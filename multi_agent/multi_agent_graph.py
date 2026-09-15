"""
Multi-Agent 协作系统 — Supervisor + Researcher + Writer。

三个 Agent 各司其职：

  Supervisor  只看状态，做调度决策："该搜了"、"该写了"、"完成了"
  Researcher  只管搜索：本地库 + 互联网，搜完把材料交出来，不管写
  Writer      只管写作：拿 Researcher 的材料，组织成用户可读的答案

为什么拆开：
  1. 搜索和写作是两个不同的认知任务，混在一起 LLM 容易顾此失彼
  2. Researcher 的 context 里全是 raw search results，Writer 的 context
     里只有整理好的材料——各看各的，不互相污染
  3. 如果以后想加 Reviewer Agent（审查答案质量），架构不用大改

图结构：

  [START] → rewrite → supervisor ←→ researcher
                          │              ↓ (搜完)
                          │           supervisor
                          │              ↓ (该写了)
                          │           writer → supervisor → END
"""
import os
import sys
import asyncio
import logging
from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain_core.tools import tool

# 长期记忆 Store
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# 记忆 checkpointer：新版 LangGraph 叫 InMemorySaver，旧版叫 MemorySaver，做兼容导入
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver

from .query_rewrite import rewrite_query
from .parent_retriever import parent_hybrid_retrieve
from .web_retriever import web_search
from .reranker import rerank, get_reranker
from .context_compressor import compress_documents

logger = logging.getLogger(__name__)


# ========== 1. 定义 State ==========

class MultiAgentState(TypedDict):
    """
    和 ToolAgentState 的区别：
    - agent_messages: 每个 Agent 有自己的消息通道，互不污染
    - research_results: Researcher 搜完后存这里，Writer 从这里取
    - next_agent: Supervisor 写入，决定下一个被调用的 Agent
    """
    question: str
    rewrite_question: str

    # Researcher 的消息通道（只有搜索工具相关的消息）
    researcher_messages: Annotated[list, operator.add]

    # Writer 的消息通道（只有研究材料 + 写作指令）
    writer_messages: Annotated[list, operator.add]

    # Researcher 搜完的成果，Writer 的输入
    research_results: str

    # Supervisor 的控制信号
    next_agent: str  # "researcher" | "writer" | "finish"

    researcher_rounds: int  # ← 加，Researcher 搜了几轮
    max_tool_rounds: int  # ← 加，上限

    # Reviewer 的通道：审核结论 + 轮数控制
    review_messages: Annotated[list, operator.add]  # 审核过程记录
    review_verdict: str      # "PASS" | "REVISE"
    review_feedback: str     # 打回时给 Researcher 的具体修改意见
    review_rounds: int       # 已审核几轮
    max_review_rounds: int   # 审核上限（防死循环）

# ========== 2. 制造搜索工具 ==========

def make_search_tools(vector_store, bm25, chunks, parent_docs, extra_tools=None):
    """
    只给 Researcher 用的搜索工具，和之前 tool_agent_graph 里的
    local_search / internet_search 逻辑一样。
    不包含 think——Researcher 只管搜，不需要停下来反思。
    """

    @tool
    def local_search(query: str) -> str:
        """
        搜索本地 AI/ML 知识库。

        适用场景：技术概念解释、框架使用方法、算法原理、架构设计。
        不适用场景：实时新闻、最新动态、产品价格。

        Args:
            query: 中文搜索查询词，要具体
        """
        results = parent_hybrid_retrieve(
            query,
            child_vector_store=vector_store,
            child_bm25=bm25,
            child_chunks=chunks,
            parent_docs=parent_docs,
            top_k=5,
        )

        if not results:
            return "本地知识库中未找到相关文档。"

        docs_for_rerank = [r["doc"] for r in results]
        reranked = rerank(query, docs_for_rerank)[:3]
        compressed = compress_documents(query, reranked, get_reranker(), keep_ratio=0.5)

        if not compressed:
            return "检索到了文档但压缩后无可保留内容。"

        contexts = []
        for i, item in enumerate(compressed, 1):
            content = item["doc"].page_content
            source = item["doc"].metadata.get("source", "未知")
            score = item.get("score", 0)
            contexts.append(f"[文档{i} | 来源={source} | 相关度={score:.3f}]\n{content}")

        return "\n\n---\n\n".join(contexts)

    @tool
    def internet_search(query: str) -> str:
        """
        搜索互联网获取实时信息。

        Args:
            query: 搜索查询词
        """
        results = web_search(query, k=5)

        if not results:
            return "未找到相关网络信息。"

        contexts = []
        for i, item in enumerate(results, 1):
            content = item["doc"].page_content
            url = item["doc"].metadata.get("source", "未知")
            contexts.append(f"[网页{i}] {url}\n{content}")

        return "\n\n---\n\n".join(contexts)

    tools = [local_search, internet_search]
    if extra_tools:
        tools += list(extra_tools)  # MCP 工具并进来，Researcher 就能看见了
    return tools
    # return [local_search, internet_search]


# ========== MCP 客户端全局复用 ==========
# 企业级要点：客户端进程全局只起一次，所有会话共享。
# 之前每个会话（每次上传 PDF / 会话恢复）都新建 MultiServerMCPClient，
# 等于每个用户都 fork 一个 Python 子进程 —— 100 个会话就 100 个进程。
_mcp_ctx = None                # 全局唯一的 MCP 上下文（client + session + connection）
_mcp_ctx_lock = asyncio.Lock()  # 并发下只初始化一次


async def _get_mcp_ctx():
    """
    返回全局唯一的 MCP 上下文，含一个常驻 ClientSession（= 一个 stdio 子进程）。

    langchain-mcp-adapters 0.1.0 之后，get_tools() 每次调用都会临时 create_session
    起新进程，工具调用时也按需重连 —— 每查一次就 fork 好几个 Python 子进程。
    只有把同一个 session 实例传给 load_mcp_tools(session=...)，工具调用才复用这条
    常驻 stdio 连接。所以这里手动 __aenter__ 长期持有 session 和它的 context
    manager（cm 留着，app 关停时 __aexit__ 干净退出）—— 整个应用生命周期只
    fork 一个 MCP 子进程，100 个会话也不会起 100 个进程。

    注意：不要用 MultiServerMCPClient.session() 包一层 —— 实测它包 callbacks 后
    首个 list_tools 会卡死（memory stream WouldBlock）。直接 create_session 直连。
    """
    global _mcp_ctx
    if _mcp_ctx is not None:
        return _mcp_ctx
    async with _mcp_ctx_lock:   # 双重检查：多个协程同时进来也只起一个进程
        if _mcp_ctx is None:
            from langchain_mcp_adapters.client import create_session

            # 关键：文件在 multi_agent/ 目录，往上一级才是项目根，再进 mcp_tools/
            server_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "mcp_tools", "my_mcp_server.py",
            )
            connection = {
                "transport": "stdio",          #通信模式，标准输入输出子进程通信
                "command": sys.executable,   # 用 .venv 的 python，保证子进程有 mcp 包
                "args": [server_path],       # 绝对路径，在哪启动 uvicorn 都不怕
            }
            cm = create_session(connection)          # 手动持有 async context manager
            session = await cm.__aenter__()            #正式拉起 MCP Server 子进程
            await session.initialize()               # 必须先初始化才能收发请求
            _mcp_ctx = {"session": session, "connection": connection, "cm": cm}
            #`session`：MCP 会话对象（最核心，传给`load_mcp_tools(session=session)`实现连接复用）
            #`connection`：连接配置（备用）
            #`cm`：上下文管理器对象，**关闭服务时必须依靠 cm 调用`__aexit__`，杀掉 MCP 子进程**。
    return _mcp_ctx #初始化完成，返回全局 MCP 上下文字典。


async def close_mcp():
    """关闭全局 MCP 连接（FastAPI 关停时调用，避免挂在 asyncio 清理的报错噪音）。"""
    global _mcp_ctx
    if _mcp_ctx is not None:
        try:
            await _mcp_ctx["cm"].__aexit__(None, None, None)
        except Exception as e:
            logger.warning("[MCP] 关闭连接时异常（可忽略）：%s", e)
        _mcp_ctx = None


def mcp_status() -> dict:
    """给健康检查用：只读地报告全局 MCP 子进程状态，不触发初始化。

    特意不做主动 ping：探活接口不该有副作用——不能因为被探活就把子进程拉起来。
    所以这里只回答"有没有常驻会话"。MCP 是可选依赖：真连不上时 Researcher
    会退回纯本地检索工具，主流程不受影响，所以它挂了只算降级、不算故障。
    """
    if _mcp_ctx is None:
        return {"status": "not_started"}   # 惰性启动：还没人上传过 PDF，属正常
    if _mcp_ctx.get("session") is None:
        return {"status": "error", "detail": "MCP 上下文存在但 session 为空"}
    return {"status": "connected"}


async def load_mcp_tools(domain: str = "general"):
    """
    加载指定领域允许使用的 MCP 工具（LangChain 工具列表）。

    架构：MCP server 暴露全部领域工具，这里按 registry 白名单过滤，
    只把该领域该用的工具交给 Researcher —— 避免 LLM 在一堆无关工具里选。

    企业级要点：传入共享 session，工具调用走同一条常驻 stdio 连接，
    不再每次调用都 fork 子进程。

    失败降级：MCP 连不上返回空列表，RAG 主流程不受影响（有日志可观测）。
    """
    try:
        # 白名单过滤，先于启动子进程：没有工具就根本不用连 MCP
        from mcp_tools.registry import get_domain_tools

        whitelist = get_domain_tools(domain)#根据领域名称取出工具白名单列表。
        if not whitelist:
            return []

        ctx = await _get_mcp_ctx()
        from langchain_mcp_adapters.client import load_mcp_tools as _load_from_session #导入适配器加载函数，起别名

        tools = await _load_from_session(
            ctx["session"], connection=ctx["connection"], server_name="mytools"
        )
        filtered = [t for t in tools if t.name in whitelist]#列表推导式做白名单过滤
        logger.info("[MCP] 领域 %s → 已加载 %d 个工具: %s", domain, len(filtered), [t.name for t in filtered])
        return filtered
    except Exception as e:
        logger.warning("[MCP] 加载 MCP 工具失败，已跳过：%s", e)
        return []   # MCP 挂了不影响 RAG 主流程



def trim_messages(messages, keep=10):
    """
    上下文截断：只保留 SystemMessage + 最近 keep 条消息。

    为什么 SystemMessage 单独留：
      SystemMessage 是"人设/任务书"，永远是列表开头，不能被截掉，
      否则 LLM 会忘记自己是谁、要干嘛。
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]#`isinstance(m, SystemMessage)` 判断这条消息是不是**系统提示词**。
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    rest = rest[-keep:]
    # 关键修复：切片可能把「tool_calls 的 AI 消息」切掉，留下孤立的 ToolMessage。
    # API 规定 tool 消息前面必须紧跟带 tool_calls 的 AI 消息，否则报 400。
    while rest and isinstance(rest[0], ToolMessage):
        rest = rest[1:]
    return system_msgs + rest[-keep:]


# ========== 3. 各节点函数 ==========

def rewrite_query_node(state: MultiAgentState, llm,config:RunnableConfig,store:BaseStore, mcp_tool_guide: str = ""):
    """
    问题改写 — 和之前完全一样。
    唯一的变化：初始化 researcher_messages 和 writer_messages 为空。
    mcp_tool_guide：构图时算好的专业工具引导文案，拼进 Researcher 提示词。
    """
    question = state["question"]
    rewritten = rewrite_query(question, llm)

    # 长期记忆：用户说"记住xxx"就把偏好存进 Store
    user_id = config.get("configurable", {}).get("user_id", "anonymous")#拿到当前聊天是谁，区分不同人的记忆，A 用户记住的东西不会跑到 B 用户身上。
    if "记住" in question:
        pref = question.split("记住", 1)[-1].strip("：:，。、 ")#从字符串只分割 1 次。`[-1]`：取分割后**最后那一段**，也就是 “记住” 后面所有文字
        if pref:#`("users", user_id)` → namespace 命名空间，相当于文件夹：users 目录下，该用户的记忆。`"style"` → key，记忆条目的名字，代表「回答风格偏好」。`{"data": pref}` → 要保存的值，必须是字典格式
            # 这里**故意**保持同步的 store.put，不是漏改 —— 和 writer 那边相反。
            # 本节点是同步函数（见下面的 def），LangGraph 会把同步节点丢进线程池执行，
            # 那条线程上没有运行中的事件循环，_check_loop 只拦"当前 loop 就是 store
            # 自己那个 loop"的情况（store/base/batch.py:33），所以这里放行，
            # run_coroutine_threadsafe 借用主循环把写入跑完，语义正确。
            # 把它改成 await store.aput(...) 反而要先把节点改成 async，那样里面
            # rewrite_query(question, llm) 这个阻塞 LLM 调用就落到事件循环上了 ——
            # 得不偿失（项目对"阻塞活别占事件循环"的要求见 tests/test_async_offload.py）。
            store.put(("users", user_id), "style", {"data": pref})
            logger.info("[Memory] 已记住用户偏好：%s", pref[:80])

    # mcp_tool_guide 由 build_multi_agent_graph 构图时算好传入（见 build 函数）
    researcher_system = SystemMessage(content=f"""你是一个专职的信息研究员。
你的唯一任务：根据 Supervisor 给你的搜索指令，调用搜索工具获取信息。

工作原则：
- 只做搜索，不做分析。搜到原始材料后直接返回，不要自己组织答案。
- 如果 Supervisor 说搜本地库，就用 local_search。
- 如果 Supervisor 说搜互联网，就用 internet_search。
- 你可以一次调用多个工具来覆盖不同角度的信息。
- 当你认为搜索已经足够，不要自己回答用户——把你的搜索结果
  以「研究材料」的形式整理好，等待 Writer 接手。
{mcp_tool_guide}
记住：你只管"找到什么"，不管"怎么说"。""")

    researcher_human = HumanMessage(
        content=f"搜索指令：请针对以下主题进行全面搜索。\n\n"
                f"用户原始问题：{question}\n\n"
                f"搜索参考方向：{rewritten}\n\n"
                f"请先搜索本地知识库（技术概念和原理），如果本地信息不够或涉及实时信息，"
                f"再搜互联网。"
                f"{'如有开源项目、论文、模型选型相关需求，优先用对应的专业检索工具。' if mcp_tool_guide else ''}"
                f"搜完后整理好材料，等待 Writer 接手。"
    )

    return {
        "rewrite_question": rewritten,
        "researcher_messages": [researcher_system, researcher_human],
        "writer_messages": [],  # Writer 还没开始，先空着
        "next_agent": "researcher",  # Supervisor 先去 Researcher
    }


def supervisor_node(state: MultiAgentState) -> dict:
    """
    Supervisor — 只看状态，不做推理。

    决策逻辑：
    1. 刚开始？→ 派 Researcher 去搜
    2. Researcher 刚刚搜完？→ 把材料交给 Writer
    3. Writer 刚写完？→ 结束
    """
    next_agent = state.get("next_agent", "researcher")

    if next_agent == "researcher":
        logger.info("[Supervisor] → 派 Researcher 去搜索")
        return {"next_agent": "researcher"}

    elif next_agent == "writer":
        # 打包 Researcher 的成果，交给 Writer
        logger.info("[Supervisor] → Researcher 搜完了，派 Writer 来写")



        research_messages = state.get("researcher_messages", [])
        research_content = ""#初始化字符串
        for msg in reversed(research_messages):
            if isinstance(msg, AIMessage):
                has_tool_calls = getattr(msg, "tool_calls", None)
                has_content = bool(msg.content)
                # 两条同时满足：有文字内容，且没有工具调用——才是真正的"收手总结"
                if has_content and not has_tool_calls:
                    research_content = msg.content
                    break

        # 构建 Writer 的初始消息
        writer_system = SystemMessage(content="""你是一个专业的技术文档写作者。
你的任务：基于 Researcher 提供的研究材料，撰写清晰、准确、结构化的回答。

工作原则：
- 所有事实必须来自研究材料，不要编造
- 如果研究材料之间有矛盾，如实指出，优先采用较新信息
- 使用标题、段落、对比表格等结构让回答易读
- 如果材料不足以回答用户问题，明确说明
- 直接输出最终回答，不需要进一步搜索""")

        writer_human = HumanMessage(
            content=f"用户问题：{state['question']}\n\n"
                    f"研究材料：\n\n{research_content}\n\n"
                    f"请基于以上材料撰写最终回答。"
        )

        return {
            "next_agent": "writer",
            "research_results": research_content,
            "writer_messages": [writer_system, writer_human],
        }

    else:  # finish
        logger.info("[Supervisor] → 任务完成")
        return {"next_agent": "finish"}


def researcher_agent_node(state: MultiAgentState, researcher_llm):
    """
    Researcher Agent — 搜索并产出研究材料。

    有工具调用 → 继续在 researcher 里循环
    无工具调用 → 搜索结束，通知 Supervisor 该 Writer 上场了
    """
    messages = state.get("researcher_messages", [])
    messages = trim_messages(messages, keep=10)
    response = researcher_llm.invoke(messages)

    rounds = state.get("researcher_rounds", 0)
    # 判断 Researcher 是不是收手了
    if getattr(response, "tool_calls", None):
        max_rounds = state.get("max_tool_rounds", 5)
        if rounds >= max_rounds:
            logger.warning("[Researcher] 已达最大搜索轮数 %d，强制切换到 Writer", max_rounds)

            # ★ 关键：强制追加一条 HumanMessage，要求总结
            summary_prompt = HumanMessage(
                content="已达到最大搜索轮数，请不要再搜索。立即将你已获得的所有搜索结果整理成一份结构化的研究材料，供 Writer 使用。"
            )
            # 用不带工具的 llm 做最后一次总结（不再能调工具了）
            summary_llm = researcher_llm.bind_tools([]) if hasattr(researcher_llm, 'bind_tools') else researcher_llm
            #hasattr(researcher_llm, 'bind_tools')判断当前 LLM 实例是否具备绑定工具的方法；部分封装后的大模型没有该接口，需要做兼容。
            #researcher_llm.bind_tools([])绑定空的工具列表.绑定空列表之后 LLM 工具池为空，**再也无法输出 tool_calls、无法调用搜索工具**
            #三元表达式：支持 bind_tools 就生成禁用工具的临时 llm；不支持则沿用原模型
            # ★ 修复：response 是带 tool_calls 的 AI 消息，后面却没有对应的 ToolMessage，
            # 直接拼进 summary_llm 会触发 400「tool_calls must be followed by tool messages」。
            # 强制停手时这条 tool_calls 不会再被执行，直接丢弃，只保留历史 + 总结指令。
            final_response = summary_llm.invoke(messages + [summary_prompt])


            return {
                "researcher_messages": [summary_prompt, final_response],
                "researcher_rounds": rounds + 1,
                "next_agent": "writer",
            }
        return {
            "researcher_messages": [response],
            "researcher_rounds": rounds + 1,
            "next_agent": "researcher",
        }
    else:
        return {
            "researcher_messages": [response],
            "researcher_rounds": rounds,
            "next_agent": "writer",
        }

async def writer_agent_node(state: MultiAgentState, llm,config:RunnableConfig,store:BaseStore):
    """
    Writer Agent — 用 Researcher 的材料生成最终答案。

    改成流式生成（async + astream）：
      - 图以 stream_mode="messages" 跑时，Writer 生成中的每个 token 会被
        LangGraph 捕获，后端能逐字推给前端（打字机效果）
      - 不用流式模式时（如评估的 ainvoke），照常累积成完整答案，行为不变
    """
    messages = state.get("writer_messages", [])
    messages = trim_messages(messages, keep=6)

    # 长期记忆：读用户偏好，注入为一条 system 消息（插在最前）
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    #
    # 必须是 aget 不能是 get —— 这个节点是 async 的，跑在事件循环线程上，
    # 而 store 若是 AIOMySQLStore，同步 get 内部是
    # `run_coroutine_threadsafe(self.aget(...), store._loop).result()`：
    # 在事件循环线程里等"丢给同一个循环的协程"，循环被自己堵死，任务永远排不上，
    # 直接死锁。langgraph 的 _check_loop 就是拦这个的（store/base/batch.py:33），
    # 抛 InvalidStateError 而不是让你挂住。
    #
    # 实测：Step 1-3 把 store 从 InMemoryStore 换成 AIOMySQLStore 之后，这个同步调用
    # 没被任何单测碰到（单测走的是内存 store，同步调用完全正常），直到容器里跑
    # 第一次真实对话才炸：SSE 推回 error 事件、这一轮没有答案。
    #
    # InMemoryStore 也有 aget（BaseStore 的默认实现，走 abatch），所以降级到内存
    # store 时同样安全 —— 改完两条路都对。
    pref = await store.aget(("users", user_id), "style")
    if pref is not None and pref.value.get("data"):
        messages = [SystemMessage(content=f"用户偏好：{pref.value['data']}。请严格按此偏好组织回答。")] + messages

    # 流式生成：config 必须透传，LangGraph 的 messages 流模式靠它捕获 token
    chunks = []
    async for chunk in llm.astream(messages, config=config):
        chunks.append(chunk)

    # 把分片拼成完整响应（AIMessageChunk 支持 += 合并）
    response = chunks[0] if chunks else AIMessage(content="")
    for c in chunks[1:]:
        response += c

    return {
        "writer_messages": [response],
        "next_agent": "finish",
    }


# ========== 3.5 Reviewer 质量审核节点 ==========

reviewer_system = SystemMessage(content="""你是一个严格的答案质量评审员（Reviewer）。

你的任务：审查 Writer 的回答是否合格，只输出结论和修改意见。

评审规则：
1. 忠实度：回答中的事实主张必须能在研究材料中找到依据；编造材料里没有的内容，直接判 REVISE。
2. 完整性：回答是否完整覆盖了用户问题的核心，有没有漏掉关键点。
3. 相关性：回答是否直接回应问题，有没有答非所问。

输出格式（严格两行，不要其他内容）：
VERDICT: PASS   或   VERDICT: REVISE
FEEDBACK: PASS 就写"无需修改"；REVISE 就写具体修改意见（缺什么信息、哪里不忠实、建议再搜什么方向）。""")


def reviewer_agent_node(state: MultiAgentState, llm, max_review_rounds=2):
    """
    Reviewer Agent — 审核 Writer 的回答。
    合格（PASS）→ 结束；不合格（REVISE）且未超上限 → 带修改意见打回 Researcher 重搜。
    """
    # 取 Writer 的最终回答
    writer_messages = state.get("writer_messages", [])
    answer = ""
    for msg in reversed(writer_messages):
        if isinstance(msg, AIMessage) and msg.content:
            answer = msg.content
            break
    if not answer:
        # 拿不到回答就放行，避免无意义的死循环
        return {"review_verdict": "PASS", "review_feedback": "（未提取到回答，跳过审核）", "next_agent": "finish"}

    research_materials = state.get("research_results", "")
    question = state["question"]

    review_human = HumanMessage(
        content=f"用户问题：{question}\n\n研究材料：\n{research_materials}\n\nWriter 的回答：\n{answer}"
    )
    response = llm.invoke([reviewer_system, review_human])

    # 解析 "VERDICT: xxx" / "FEEDBACK: xxx" 两行，解析失败默认 REVISE（宁严勿松）
    text = response.content.strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    verdict, feedback = "REVISE", text
    for ln in lines:
        upper = ln.upper()
        if upper.startswith("VERDICT:"):
            verdict = "PASS" if "PASS" in upper else "REVISE"
        elif upper.startswith("FEEDBACK:"):
            feedback = ln.split(":", 1)[1].strip()

    rounds = state.get("review_rounds", 0)
    max_rounds = state.get("max_review_rounds", max_review_rounds)

    logger.info("[Reviewer] 第 %d 轮审核 → %s", rounds + 1, verdict)

    # 不合格且未超上限：把修改意见作为一条 HumanMessage 追加给 Researcher，让它针对性重搜
    if verdict == "REVISE" and rounds < max_rounds:
        feedback_msg = HumanMessage(
            content=f"【Reviewer 修改意见】{feedback}\n请根据以上意见，补充搜索缺失的信息，然后整理成新的研究材料。"
        )
        return {
            "review_messages": [AIMessage(content=text)],
            "review_verdict": verdict,
            "review_feedback": feedback,
            "review_rounds": rounds + 1,
            "researcher_messages": [feedback_msg],
            "next_agent": "researcher",
        }

    # 通过或已达上限：结束
    return {
        "review_messages": [AIMessage(content=text)],
        "review_verdict": verdict,
        "review_feedback": feedback,
        "review_rounds": rounds + 1,
        "next_agent": "finish",
    }


# ========== 4. 路由函数 ==========

def route_supervisor(state: MultiAgentState) -> str:
    """Supervisor 之后该谁上场"""
    next_agent = state.get("next_agent", "finish")
    if next_agent == "researcher":
        return "researcher_agent"
    elif next_agent == "writer":
        return "writer_agent"
    else:
        return END


def route_researcher(state: MultiAgentState) -> str:
    """
    Researcher 输出后：
    - 如果要继续搜 → 回 researcher_tool_node
    - 如果搜完了 → 回 supervisor，让它重判
    """
    next_agent = state.get("next_agent", "writer")
    if next_agent == "researcher":
        return "researcher_tool_node"
    else:
        return "supervisor_node"


def route_reviewer(state: MultiAgentState) -> str:
    """
    Reviewer 审核后：
    - REVISE 且未超上限 → 打回 researcher_agent 重搜（带上修改意见）
    - PASS 或已达上限 → 结束
    """
    next_agent = state.get("next_agent", "finish")
    if next_agent == "researcher":
        return "researcher_agent"
    else:
        return END


async def researcher_tool_execute(state, researcher_tool_node):
    """执行 Researcher 的工具调用 + 日志。

    ★ 必须是 async：MCP 工具是纯异步工具（只有 coroutine 没有 func），
      同步 invoke 会抛 NotImplementedError("StructuredTool does not support sync invocation")，
      所以这里必须走 ainvoke。
    """
    messages = state.get("researcher_messages", [])
    last_message = messages[-1] if messages else None
    tool_calls = getattr(last_message, "tool_calls", [])

    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("args", {})
        short_args = {}
        for k, v in args.items():
            s = str(v)
            short_args[k] = s[:80] + "..." if len(s) > 80 else s
        logger.info("[Researcher] %s(%s)", name, short_args)

    result = await researcher_tool_node.ainvoke({"messages": messages})
    """
    1. ToolNode 内部工作流程：
   - 读取 AIMessage 中的 tool_calls
   - 执行对应搜索函数  ,local_search函数这里利用了闭包：
            make_search_tools 执行时，内部的 local_search 函数定义"捕获"了外层作用域里的四个变量。
            即使 make_search_tools 已经 return，这四个值依然被 local_search 拽着不放。
            等到 ToolNode 调用 local_search(query="RAG原理") 时，
            query 从参数拿，另外四个从闭包里拿——外部调用者只看到一个干净的 (query) 接口，内部的五个依赖却一个不少。
   - 生成 ToolMessage（工具返回结果消息）
   - 返回全新、追加完 ToolMessage 的 messages 列表
    2.result 是字典，key 为 "messages"
    """
    return {
        "researcher_messages": result["messages"],
        "next_agent": "researcher",  # 工具执行完，让 Researcher Agent 再看结果
    }


# ========== 5. 构建图 ==========

def build_multi_agent_graph(
    llm,
    vector_store,
    bm25,
    chunks,
    parent_docs,
    max_tool_rounds=5,
    max_review_rounds=2,   # ← 新增：Reviewer 最多打回几轮，防死循环
    checkpointer=None,# ← 新增：允许外部传入 checkpointer
    store=None,         # ← 新增：长期记忆 Store
    extra_tools=None
):
    """
    构建 Multi-Agent 协作图。

    图结构：

         [START]
           │
           ▼
      rewrite_query_node
           │
           ▼
      supervisor_node
           │
           ├── "researcher" → researcher_agent ←→ researcher_tool_node
           │                       │ (搜完)
           │                       ▼
           │                  supervisor_node
           │                       │
           ├── "writer" ───→ writer_agent
           │                       │
           │                       ▼
           │                  reviewer_agent
           │                  ┌───┴───┐
           │            PASS  │       │  REVISE(未超上限)
           │                  ▼       ▼
           │               [END]  researcher_agent（带修改意见重搜）
           │                              │
           │                              ▼
           │                         ...最终再回到 writer_agent → reviewer_agent
           └── "finish" ───→ [END]
    """
    # Researcher 的工具
    search_tools = make_search_tools(vector_store, bm25, chunks, parent_docs,extra_tools=extra_tools)
    researcher_llm = llm.bind_tools(search_tools)
    researcher_tool_node = ToolNode(search_tools)

    # MCP 领域工具引导：构图时算好，传给 rewrite_query_node 拼进 Researcher 提示词。
    # （光 bind_tools 不够——提示词里只点名 local_search/internet_search 时，
    #   LLM 会一直优先用那两个，专业工具形同虚设）
    mcp_tool_guide = ""
    if extra_tools:
        lines = []
        for t in extra_tools:
            desc = (t.description or "专业检索工具").splitlines()[0][:60]
            lines.append(f"  {t.name}：{desc}")
        mcp_tool_guide = (
            f"\n- 你还有 {len(extra_tools)} 个专业检索工具（按需使用）：\n"
            + "\n".join(lines)
            + "\n- 当问题涉及开源项目选型、学术论文、模型选型等方向时，"
              "优先用对应的专业工具，而不是只靠网页搜索。"
        )

    graph = StateGraph(MultiAgentState)

    # 注册节点
    graph.add_node("rewrite_query_node", lambda s,config,store: rewrite_query_node(s, llm,config,store, mcp_tool_guide))
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("researcher_agent", lambda s: researcher_agent_node(s, researcher_llm))
    async def _researcher_tool_node(s):
        # async 包装：让 LangGraph 通过 ainvoke 执行工具（MCP 工具同步跑会炸）
        return await researcher_tool_execute(s, researcher_tool_node)

    graph.add_node("researcher_tool_node", _researcher_tool_node)
    async def _writer_agent_node(s, config, store):
        # async 包装：writer 现在是 astream 流式生成
        return await writer_agent_node(s, llm, config, store)

    graph.add_node("writer_agent", _writer_agent_node)
    graph.add_node("reviewer_agent", lambda s: reviewer_agent_node(s, llm, max_review_rounds))

    # 连线
    graph.add_edge(START, "rewrite_query_node")
    graph.add_edge("rewrite_query_node", "supervisor_node")

    # Supervisor → Researcher / Writer / END
    graph.add_conditional_edges(
        "supervisor_node",
        route_supervisor,
        {
            "researcher_agent": "researcher_agent",
            "writer_agent": "writer_agent",
            END: END,
        }
    )

    # Researcher 内部循环：agent ↔ tool
    graph.add_conditional_edges(
        "researcher_agent",
        route_researcher,
        {
            "researcher_tool_node": "researcher_tool_node",
            "supervisor_node": "supervisor_node",
        }
    )
    graph.add_edge("researcher_tool_node", "researcher_agent")

    # Writer 写完 → Reviewer 审核（PASS 结束 / REVISE 打回 Researcher 重搜）
    graph.add_edge("writer_agent", "reviewer_agent")
    graph.add_conditional_edges(
        "reviewer_agent",
        route_reviewer,
        {
            "researcher_agent": "researcher_agent",  # REVISE：带修改意见回 Researcher 重搜
            END: END,                                # PASS / 超上限：结束
        }
    )
    if checkpointer is None:
        checkpointer = MemorySaver()  # ← 新增：默认用内存版
    if store is None:
        store=InMemoryStore()          # ← 新增：默认用内存版长期记忆
    return graph.compile(checkpointer=checkpointer,store=store)   # ← 原来是 graph.compile()


# ========== 6. 提取最终回答 ==========

def extract_answer(result: dict) -> str:
    """从 writer_messages 中提取最终回答"""
    writer_messages = result.get("writer_messages", [])
    for msg in reversed(writer_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "（未能提取到最终回答）"