"""
FastAPI 后端 — 把 Multi-Agent 系统封装成 REST API。

三个接口：
  POST /api/upload-pdf    上传 PDF，建索引，返回 session_id
  POST /api/chat-stream   流式对话（SSE），实时推送搜索状态和最终答案
  GET  /api/health        健康检查

架构关系：
  Streamlit 前端 ↔ FastAPI 后端 ↔ LangGraph Agent-RAG 核心
"""

import asyncio #事件循环里不能跑同步阻塞活：用 asyncio.to_thread 把它丢进线程池，期间事件循环继续处理别的请求
from contextlib import asynccontextmanager #lifespan 就是个 async context manager：yield 前是启动、后是关停
import os
import secrets #API Key 用 secrets.compare_digest 做常量时间比较，避免定时攻击
import shutil #用于递归清理临时目录（try/finally 保证出错也会删干净）

import uuid #生成唯一会话 id，每个上传 PDF 的用户拥有独立知识库
import json #序列化字典字符串，SSE 传输的数据必须为 JSON 字符串
import tempfile #创建操作系统临时文件夹，接收上传 PDF、加载文档，用完立刻清理，不占用磁盘
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Response #`UploadFile`：FastAPI 封装的上传文件对象，包含文件名、二进制内容。`File`：用来声明接口参数是上传文件；`Form`：接收非文件表单字段（如领域）。`Depends`：把鉴权函数挂成路由依赖；`Header`：读取请求头；`Response`：健康检查里动态改状态码
from fastapi.responses import StreamingResponse #StreamingResponse:返回流式响应，适配 SSE 长连接，可以循环 yield 不断向前端发送消息，适合 LLM 流式输出、Agent 运行日志推送
from fastapi.middleware.cors import CORSMiddleware #CORS 跨域中间件.浏览器同源策略：网页域名、端口和后端不一致就会拦截请求；你的 Streamlit 默认端口 8501，FastAPI 端口 8000，端口不同属于跨域，必须开启 CORS。
from pydantic import BaseModel, ConfigDict, field_validator #Pydantic 数据校验模型；FastAPI 依靠 BaseModel 自动校验前端传参类型、做参数解析。field_validator：给单个字段挂自定义校验规则。ConfigDict：模型级配置（这里用 extra="allow" 放行未知字段）
from typing import Literal #响应模型里把取值集合钉死（status 只能是 ok/degraded/error）
from langchain_community.document_loaders import PyPDFLoader

# 先于任何 multi_agent 模块导入配置日志：config.py 等 import 时就会打日志
from multi_agent.logging_setup import setup_logging
setup_logging()
import logging

logger = logging.getLogger(__name__)

from multi_agent.parent_splitter import split_parent_child
from multi_agent.embedding import get_embeddings
from multi_agent.vector_db import get_vector_store
from multi_agent.bm25 import create_bm25
from multi_agent.llm import get_llm
from multi_agent.multi_agent_graph import build_multi_agent_graph, extract_answer, load_mcp_tools, mcp_status, close_mcp

from langgraph.checkpoint.mysql.aio import AIOMySQLSaver
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()  # 全局唯一长期记忆，跨会话共享（内存版，后端重启即清空）
# ========== 持久化记忆：MySQL checkpointer ==========
# 对话记忆（checkpoint）落到 MySQL，后端重启不丢、多进程共享同一份。
#
# 这里只留一个占位，真正的连接池在 lifespan 启动时才建：建池要 await，
# 而模块 import 发生在还没有事件循环的时刻，建不了。
# 值为 None 时 build_multi_agent_graph 内部会自动退回 MemorySaver —— 记忆退化成
# 进程内存（重启即清空），问答本身照跑。这和 /api/health 把 MySQL 算作软依赖
# 是同一个判断：连不上库不该让整个服务起不来。
checkpointer = None

# 注意上面这行 import 特意没包 try/except ImportError。
# 原来那版 SqliteSaver 是包了的（"没装就退回内存版，不报错"），结果 langgraph-
# checkpoint-sqlite 从头到尾就没装进依赖里，ImportError 被静默吞掉，于是对话记忆
# 从来没有真正落过盘（checkpoints.sqlite 一直是 0 字节），而且没有任何症状。
# 缺依赖是打包错误，不是运行期状况，就该在启动时响亮地炸掉。
# （包本身可以裸 import：langgraph/checkpoint/mysql/__init__.py 不导驱动，
#  真正需要 aiomysql 的只有 .aio 子模块，而 aiomysql 已在 requirements 里。）


async def _init_checkpointer():
    """建连接池 + 建表，返回 (saver, pool)；任何失败都返回 (None, None) 并记日志。

    抽成独立函数两个原因：
      · lifespan 里只剩三行，启动/关停的资源一眼能看全；
      · 单测能整体替换掉它 —— 否则每个跑 lifespan 的用例都得去连一个真 MySQL。

    失败不往外抛：MySQL 是软依赖（判定见 /api/health），连不上只该让记忆退化成
    进程内存，不该让整个服务起不来。但也不能一声不吭 —— 记 warning，让"记忆没持久化"
    这件事在日志里留痕。
    """
    pool = None
    try:
        pool = await create_mysql_pool()
        # 官方入口是 AIOMySQLSaver.from_conn_string(DSN)，这里走的是
        # AIOMySQLSaver(conn=pool) —— 这条路没写在文档里，靠的是包内部对连接的
        # 鸭子类型判断（_ainternal.get_connection：有 acquire 当池用，有 cursor
        # 当单连接用）。所以它不是公开 API，升级 langgraph-checkpoint-mysql 时
        # 要重新确认一遍。
        # 换来的东西值这个风险：绕开 DSN 的密码转义与字符集两个静默坑，
        # 且没有单连接 8 小时空闲被服务端掐断的问题（理由见 db.create_mysql_pool）。
        saver = AIOMySQLSaver(conn=pool)
        # 建表是 CREATE TABLE IF NOT EXISTS，幂等，每次启动都可以安全跑一遍。
        # 它跑的是内置的迁移脚本，顺带会把 schema 升到当前包版本要求的样子。
        #
        # 每次启动都会看到一行 SQL 告警 "Table 'checkpoint_migrations' already
        # exists"（MySQL 1050），由 aiomysql 的 cursor 转成 Python warning 打到
        # stderr。这是 IF NOT EXISTS 命中已有表的正常结果，不是错误 —— 只有第一次
        # 启动的库才不会有。这里刻意不去 filter 掉它：告警文案随 MySQL 版本变，
        # 按文案正则过滤很脆，而且会把 setup() 真正该报的 DDL 告警一起吞掉。
        await saver.setup()
    except Exception as e:
        logger.warning(
            "MySQL 初始化失败，对话记忆退回进程内存（重启即清空）：%s: %s",
            type(e).__name__, e,
        )
        if pool is not None:
            pool.close()
            await pool.wait_closed()
        return None, None
    return saver, pool


# ========== 应用生命周期 ==========
# 原先用 @app.on_event("shutdown")，FastAPI 已弃用这套（pytest 里会打
# DeprecationWarning，指向文档的 Lifespan Events）。lifespan 是它的正式替代：
# 一个 async context manager，yield 之前是启动、之后是关停。
#
# 相比 on_event 的实际好处：启动和关停写在同一个函数作用域里，启动时打开的资源
# 可以直接用局部变量传给关停段；on_event 得拆成两个互不相干的函数，中间只能靠
# 模块级全局变量传递，共享的东西一多就散。另外 lifespan 能拿到 app 实例（on_event
# 的回调不接参数），要往 app.state 上挂东西也只能用它。
#
# 注意定义顺序：lifespan 必须在 FastAPI(...) 之前定义好并作为参数传进去，
# 这正是 on_event 装饰器写法最容易踩的坑——它是"后注册"，所以能写在 app 之后。
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 这个 global 不能省：checkpointer 是模块级那一个，/api/chat-stream 建图、
    # get_session 重建会话、/api/health 报状态三处都读它。不声明的话下面那行只是
    # 在函数内新建一个同名局部变量，外面永远看不到 —— 症状是"池都建好了、
    # 日志也说连上了，记忆却还是重启就丢"，很难往作用域上想。
    global checkpointer

    # ---- 启动 ----
    # 留这行日志是为了在日志里标出进程重启的边界——排查"重启后第一次提问
    # 为什么慢"时能一眼看到分界。
    logger.info("后端启动：接口 %s", "已鉴权" if API_KEY else "无鉴权（本地开发）")
    # 建连接池 + 建表。放在这里而不是 import 时，是因为它要 await，而模块 import
    # 发生在还没有事件循环的时刻。MCP 仍是懒加载的（等第一个带 domain 的会话
    # 才拉起子进程），这里不等它。
    checkpointer, _mysql_pool = await _init_checkpointer()
    # 排查"重启后对话记忆还在不在"的第一现场：说 MySQL 就是持久化的；说进程内存
    # 就说明上面那步失败了（具体原因在紧随其前的 warning 里）。
    logger.info(
        "对话记忆：%s", "MySQL（重启不丢）" if checkpointer else "进程内存（重启即清空）"
    )
    yield
    # ---- 关停 ----
    # 先收连接池：close() 是同步方法，只负责标记关闭、唤醒还等在 acquire 上的协程；
    # 真正的"连接都还回来了"要 await wait_closed()。少了它会留一批连接挂在
    # MySQL 侧，要等服务端自己的超时才清掉。
    if _mysql_pool is not None:
        _mysql_pool.close()
        await _mysql_pool.wait_closed()
    # 置回 None：池已经关了，再留着这个 saver 就是指向已关闭资源的悬空引用。
    # （单测里同一个进程会多次进出 lifespan，不复位的话第二次跑就开始报
    # "Cannot acquire connection after closing pool"，而错误现场在 saver 里，
    # 看不出根因是上一轮没清干净。）
    checkpointer = None
    # 关掉全局 MCP 子进程。不关的话它会挂在 asyncio 的事件循环清理上，
    # 退出时刷一堆噪音错误，盖掉真正的报错。
    await close_mcp()
    logger.info("后端已关停")


app = FastAPI(title="Adaptive Research Agent API", lifespan=lifespan)

# CORS：允许 Streamlit 前端（8501 端口）跨域调用后端（8000 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    # 只放行 Streamlit 前端来源。注意：allow_origins=["*"] 与 allow_credentials=True
    # 是浏览器禁止的组合（通配符不可带凭证），必须写死具体 origin。
    allow_credentials=True,#允许携带 cookie 凭证
    allow_methods=["*"],#允许全部 HTTP 方法 GET/POST/PUT/DELETE
    allow_headers=["*"],#允许前端所有请求头
)

# ========== 鉴权 ==========
# CORS 不是认证：它只约束浏览器，curl / requests 直接打 :8000 完全绕开。
# 不设防的话谁都能上传 PDF、跑 Agent、烧你的 token —— 所以加一层 API Key。
# 配置：.env 里设 BACKEND_API_KEY=xxx（前端读同一变量，见 app.py）；
#       docker-compose 已把它透传给两个容器。
# 没设怎么办：按开发模式放行，但启动时打一条 WARNING。
#   为什么不直接启动失败：这个项目要能 docker-compose up 就起来给面试官点开。
#   但警告必须显式——绝不能让人"以为有鉴权其实没有"。
API_KEY = os.getenv("BACKEND_API_KEY", "").strip()
if not API_KEY:
    logger.warning("未设置 BACKEND_API_KEY —— 接口无鉴权，仅限本地开发；部署环境务必设置")


async def verify_api_key(
    authorization: str = Header(default=""),
    x_api_key: str = Header(default=""),
):
    """校验 API Key。同时接受 `Authorization: Bearer <key>` 和 `X-API-Key: <key>`。

    健康检查故意不挂这个依赖：探活（k8s / 负载均衡器）不会带凭证。
    """
    if not API_KEY:            # 开发模式：没配置就不校验
        return

    token = ""
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()

    # 用 compare_digest 而不是 ==：== 在第一个不同字符处就返回，
    # 攻击者能靠响应时间逐字节试出 Key（定时攻击）。compare_digest 是常量时间。
    if not token or not secrets.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="缺少或错误的 API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ========== 会话管理 ==========
# 每个用户（session_id）对应一套独立的知识库和 graph
# 生产环境用 Redis —— session_store 把重物落盘 + 元数据进 Redis，重启不丢、多进程共享。
# 内存里只留最近用过的 graph 缓存（session_store 内部 LRU，最多 20 个）。
from streamlit_1.session_store import (
    save_session, get_session, delete_session, redis_status, SESSIONS_DIR,
)
from streamlit_1.db import mysql_status, create_mysql_pool


# ========== 请求/响应模型 ==========
# 响应模型不只是文档：声明了它，FastAPI 会把返回值按模型校验一遍再序列化，
# 于是"字段名拼错 / 类型写错"这类静默 bug 会当场变成 500，而不是让前端
# 悄悄拿到 undefined。代价是多一次 序列化 → 校验 → 再序列化，字段少时可忽略。

class ChatRequest(BaseModel):
    session_id: str
    question: str
    max_tool_rounds: int = 5
    # 长期记忆的隔离键：图里用它做 Store 命名空间 ("users", user_id)，
    # 不同 user_id 的偏好互不可见（multi_agent_graph.py 里读、写各一处）。
    # 故意不给默认值：一旦有默认值，"忘了传"的调用方就会悄悄退回共用命名空间，
    # 也就是之前写死 "21702" 那个「所有人共享一份记忆」的老 bug。
    user_id: str

    @field_validator("user_id")
    @classmethod
    def _check_user_id(cls, v: str) -> str:
        # 这个值会被直接当成 Store 的命名空间键：留着首尾空格会造出
        # "alice" 和 "alice " 两个看起来一样、实际互不相通的用户。
        v = v.strip()
        if not v:
            raise ValueError("user_id 不能为空")
        if len(v) > 64:
            raise ValueError("user_id 不能超过 64 个字符")
        return v

    #FastAPI 接收 POST 请求 json 体依靠`BaseModel`
    #如果前端传参类型错误，FastAPI 自动返回报错，不需要手写 if 判断参数类型


class UploadResponse(BaseModel):
    """上传接口的响应。前端 app.py 就按这几个字段取值（data["child_chunks"] 等），
    所以字段名是前后端之间的契约，不是随便起的。"""
    session_id: str          # 后续对话接口必须带上它
    page_count: int          # 解析出的 PDF 总页数（不是文件数）
    child_chunks: int        # 子块数：进向量库 + BM25 的粒度
    parent_chunks: int       # 父块数：命中后真正喂给 LLM 的粒度
    domain: str              # 回显实际生效的领域（决定挂了哪些 MCP 工具）


class ComponentCheck(BaseModel):
    """单个依赖的探测结果。

    三个探测器带的字段不完全一样（redis 多带 active_sessions，session_dir 多带
    path，出错时都多带 detail），所以这里把已知字段全声明成可选——/docs 里一眼
    能看出每个组件可能报出什么。

    extra="allow" 是刻意的：以后某个探测器多返回一个字段（比如 latency_ms），
    response_model 默认会把**没声明的字段静默剥掉**——数据在 API 层凭空消失，
    且没有任何报错。allow 让未知字段原样透传，声明只管文档、不管过滤。
    """
    model_config = ConfigDict(extra="allow")

    # 取值不是一套：redis 是 ok/error，mcp 是 connected/not_started/error，
    # checkpointer 是 ok/disabled。所以这里用 str 而不是 Literal——
    # 健康检查最不该因为"某个探测词表变了"把自己变成 500。
    status: str
    detail: str | None = None
    path: str | None = None
    active_sessions: int | None = None


class HealthResponse(BaseModel):
    """整体状态。这个集合是封闭的，用 Literal 把取值钉死，
    /docs 里直接渲染成下拉可选项。"""
    status: Literal["ok", "degraded", "error"]
    checks: dict[str, ComponentCheck]     # 键是组件名，值是各自的探测结果
    active_sessions: int | None = None    # Redis 挂了就探不到，为 null 而不是编一个


# ========== 上传安全限制 ==========
MAX_FILES = 5                      # 单次最多上传文件数
MAX_FILE_SIZE = 20 * 1024 * 1024   # 单个文件最大 20MB
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 单次上传总量上限 50MB


# ========== 接口 1：上传 PDF + 建索引 ==========

@app.post("/api/upload-pdf", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
async def upload_pdf(
    files: list[UploadFile] = File(...),
    domain: str = Form("general"),
):
    """
    上传一个或多个 PDF 文件，自动切分、建向量库 + BM25、编译 Agent 图。
    domain：知识库领域（ai_learning/general/...），决定给 Researcher 挂哪些 MCP 工具。
    返回 session_id，之后的对话接口需要带上这个 ID。
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个 PDF 文件")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_FILES} 个文件")

    # 1. 读取并校验文件（类型 / 大小 / 数量），全部在内存中完成，未通过校验的文件不落盘
    file_blobs = []          # [(安全文件名, 二进制内容), ...]
    total_size = 0
    for f in files:
        # 防路径穿越：只取文件名最后一段，剥掉任何目录部分（/ 和 \ 都处理）。
        # 否则恶意文件名如 "../../../etc/x" 会写出临时目录。
        filename = f.filename.replace("\\", "/").split("/")[-1] if f.filename else "upload.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"仅支持 PDF 文件：{filename}")

        content = await f.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"文件为空：{filename}")
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"文件超过 {MAX_FILE_SIZE // (1024*1024)}MB 上限：{filename}")
        total_size += len(content)
        if total_size > MAX_TOTAL_SIZE:
            raise HTTPException(status_code=413, detail="上传总量超过上限")

        file_blobs.append((filename, content))

    # 2. 生成会话 ID
    session_id = str(uuid.uuid4())[:8]

    # 3+4. 落盘临时文件 + 解析 PDF。
    #      写盘和 PyPDF 解析都是同步阻塞（磁盘 IO + CPU），直接写在 async 函数里
    #      会把事件循环占死——上传期间别人的 /api/health、对话请求全部排队。
    #      所以整段封成同步函数丢进线程池，事件循环立刻空出来接别的请求。
    #      try/finally 保证临时目录必删，不残留垃圾文件。
    def _save_and_parse(blobs):
        temp_dir = tempfile.mkdtemp()
        try:
            saved_paths = []
            for filename, content in blobs:
                # 用 uuid 前缀重命名，彻底避免同名文件互相覆盖
                safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
                file_path = os.path.join(temp_dir, safe_name)
                with open(file_path, "wb") as f_out:
                    f_out.write(content)
                saved_paths.append(file_path)

            parsed = []
            for file_path in saved_paths:
                parsed.extend(PyPDFLoader(file_path).load())
            return parsed
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    docs = await asyncio.to_thread(_save_and_parse, file_blobs)

    try:
        # 5+6. 切分 + 建索引：这是全流程最重的同步部分——
        #      jieba 分词是纯 CPU（首次调用还要加载词典），
        #      FAISS.from_documents 更重：要为每个 chunk 同步发一次 embedding 网络请求。
        #      整段一起丢线程池，不切碎（避免多次线程切换的开销）。
        def _build_index():
            child, parent = split_parent_child(
                docs, child_size=200, parent_size=800, overlap=50
            )
            embs = get_embeddings()
            # use_cache=False：上传必须基于新文档强制重建索引，
            # 不能复用全局 faiss_db/ 里的旧索引（否则每个会话共享同一份旧库）
            vs = get_vector_store(child, embs, use_cache=False)
            return child, parent, vs, create_bm25(child)

        child_docs, parent_docs, vector_store, bm25 = await asyncio.to_thread(_build_index)

        # 7. 构建 Agent 图
        #    这一步留在线程池外：build_multi_agent_graph 只编译图结构（建节点/连边），
        #    不做 IO；真正耗时的 MCP 工具加载本身已经是异步的（await load_mcp_tools）。
        llm = get_llm()
        mcp_tools = await load_mcp_tools(domain)  # 按领域加载 MCP 工具（全局复用客户端）
        graph = build_multi_agent_graph(
            llm, vector_store, bm25, child_docs, parent_docs, max_tool_rounds=5,
            checkpointer=checkpointer,   # ← 持久化记忆；None 时内部自动退回 MemorySaver
            store=store,
            extra_tools=mcp_tools,  # ← 新增：MCP 工具接进 Researcher
        )
    except Exception as e:
        # 索引构建失败：返回 500 + 错误详情。
        # 之前直接抛 500 会让前端收到"没有 session_id 的 200"，前端会 KeyError。
        logger.exception("索引构建失败：%s: %s", type(e).__name__, e)
        raise HTTPException(status_code=500, detail=f"索引构建失败：{type(e).__name__}: {e}")

    # 8. 存进会话：重物落盘 + 元数据进 Redis + 内存缓存（三层，见 session_store）
    #    save_session 是同步函数（把 FAISS 索引 + 两个 JSON 写磁盘，再发 Redis 请求），
    #    大索引落盘是实打实的磁盘 IO，同样丢线程池，不占事件循环。
    await asyncio.to_thread(
        save_session,
        session_id,
        {
            "vector_store": vector_store,
            "bm25": bm25,
            "chunks": child_docs,
            "parent_docs": parent_docs,
            "graph": graph,
            "llm": llm,
            "domain": domain,   # 存领域，会话恢复时按它重新加载对应 MCP 工具
        },
        len(docs),
    )

    return {
        "session_id": session_id,
        "page_count": len(docs),
        "child_chunks": len(child_docs),
        "parent_chunks": len(parent_docs),
        "domain": domain,
    }


# ========== 接口 2：流式对话（核心） ==========

@app.post(
    "/api/chat-stream",
    dependencies=[Depends(verify_api_key)],
    # 这个接口没有 response_model，也不可能有：返回的是 SSE 长连接，
    # 响应体是一串不断追加的事件帧，不是"一次成形的一个 JSON 对象"，
    # response_model 那套"校验完整返回值"的前提在这里不成立。
    # 事件契约改用 responses 写进 /docs（否则文档里只有一行 text/event-stream）。
    # 注意：下面这段和函数自己 docstring 里的清单是同一份契约的两处副本，
    # 增删事件类型时两边都要改（和 app.py 里 DOMAIN_MAP 那个同步提醒同理）。
    responses={
        200: {
            "description": "SSE 流。每帧形如 `data: {...}\\n\\n`，事件类型：\n\n"
                           "- `status`：Agent 进度，带 agent/rounds\n"
                           "- `writer_start`：新一轮写作开始（Reviewer 打回时会再次出现）\n"
                           "- `token`：答案分片，逐字追加做打字机效果\n"
                           "- `tool_call`：Researcher 调了哪个工具、参数是什么\n"
                           "- `done`：终稿。**前端要用它覆盖前面流式拼出来的内容**\n"
                           "- `error`：中途失败，带 message\n",
            "content": {
                "text/event-stream": {
                    "example": 'data: {"type": "status", "agent": "researcher", "rounds": 1}\n\n'
                               'data: {"type": "token", "content": "RAG 是"}\n\n'
                               'data: {"type": "done", "answer": "RAG 是检索增强生成……"}\n\n'
                }
            },
        },
        404: {"description": "session_id 不存在（没上传 PDF，或已被清理）"},
    },
)
async def chat_stream(req: ChatRequest):
    """
    流式对话接口，返回 SSE（Server-Sent Events）。

    每条事件的格式（完整清单，/docs 里的 responses 是同一份契约）：
      {"type": "status",       "agent": "researcher", "rounds": 1}
      {"type": "status",       "agent": "writer"}
      {"type": "writer_start"}                          # Reviewer 打回重写时会再次出现
      {"type": "token",        "content": "分片"}
      {"type": "tool_call",    "name": "local_search", "args": {...}}
      {"type": "done",         "answer": "最终回答..."}  # 终稿，覆盖流式拼出的内容
      {"type": "error",        "message": "..."}
    """
    #Pydantic 模型 — ChatRequest 类（第 49-51 行）自动校验请求体。如果前端少传了 session_id 或类型不对，FastAPI 自动返回 422 错误，不需要手写校验逻辑。
    if not req.question or len(req.question) > 2000:
        raise HTTPException(status_code=400, detail="问题为空或超过 2000 字")
    # 从 Redis 取会话：内存缓存没命中 → 按 Redis 元数据从磁盘重建（async）
    session = await get_session(req.session_id, checkpointer=checkpointer, store=store)
    if not session:
        # 会话不存在返回 404（之前是 200 + {"error": ...}，前端会误判为成功）
        raise HTTPException(status_code=404, detail=f"会话 {req.session_id} 不存在，请先上传 PDF")

    graph = session["graph"]

    async def event_stream():
        result = None
        # 两个记忆维度，别搞混：
        #   thread_id = 会话 ID → 多轮对话记忆（同一 session 的多次提问串起来）
        #   user_id   = 长期记忆命名空间 → 跨会话的偏好记忆（"记住xxx"存这里）
        # user_id 由请求方提供，不再写死：写死等于所有用户共用一份偏好。
        config = {
            "configurable": {"thread_id": req.session_id, "user_id": req.user_id},
            # 给 LangSmith 用的轨迹标识：后端每轮对话生成独立 run，
            # 按 run_name=chat_stream + tag=web 就能在 smith 里筛出线上请求。
            "run_name": "chat_stream",
            "tags": ["web", "sse"],
            "metadata": {"session_id": req.session_id},
        }
        try:
            async for mode, chunk in graph.astream(
                {"question": req.question, "researcher_rounds": 0, "max_tool_rounds": req.max_tool_rounds},
                config=config,
                stream_mode=["values", "messages"],  # values=节点级状态；messages=Writer 生成 token
            ):
                if mode == "values":
                    result = chunk
                    next_agent = chunk.get("next_agent", "")
                    rounds = chunk.get("researcher_rounds", 0)

                    if next_agent == "researcher":
                        yield f"data: {json.dumps({'type': 'status', 'agent': 'researcher', 'rounds': rounds}, ensure_ascii=False)}\n\n"
                    elif next_agent == "writer":
                        # Writer 每轮开写前发一个信号，前端用它清空上一轮草稿
                        yield f"data: {json.dumps({'type': 'writer_start'}, ensure_ascii=False)}\n\n"

                    # 推送工具调用事件
                    research_msgs = chunk.get("researcher_messages", [])
                    if research_msgs:
                        last_msg = research_msgs[-1]
                        tool_calls = getattr(last_msg, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name'], 'args': tc['args']}, ensure_ascii=False)}\n\n"

                elif mode == "messages":
                    # messages 流：每个 chunk 是 (AIMessageChunk, metadata)
                    # 只透传 Writer 的生成 token；Researcher/Reviewer 的调用被过滤，不打扰前端
                    msg_chunk, metadata = chunk
                    if metadata.get("langgraph_node") == "writer_agent":
                        content = getattr(msg_chunk, "content", "")
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

            # 推送最终答案（终稿，含 Reviewer 修订后的结果）
            answer = extract_answer(result)
            yield f"data: {json.dumps({'type': 'done', 'answer': answer}, ensure_ascii=False)}\n\n"#`ensure_ascii=False`：让中文正常显示，不会被转成 `\uXXXX`
        except Exception as e:
            # 关键修复：Agent 中途出错不再直接断连，而是把错误作为 SSE 事件发回前端
            err_msg = f"{type(e).__name__}: {e}"
            logger.exception("Agent 运行出错：%s", err_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': err_msg}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
#`media_type="text/event-stream"` 声明响应体为 SSE 流式协议，浏览器识别之后开启长连接，持续监听后端推送事件


# ========== 接口 3：健康检查 ==========

@app.get(
    "/api/health",
    response_model=HealthResponse,
    # 没值的字段整个不出现，而不是铺一地 "detail": null —— 健康检查的输出
    # 经常是人在看（curl 一下），干净比完整重要。
    response_model_exclude_none=True,
    # 503 是正常业务路径（Redis 挂了就是这样），不是异常。声明进 responses
    # 才会出现在 /docs 里，调用方不用读源码就知道要处理这个状态码。
    responses={503: {"model": HealthResponse, "description": "硬依赖（Redis 或索引目录）不可用"}},
)
async def health(response: Response):
    """逐个探测依赖，如实报告是谁挂了。

    设计取舍：
      · 不鉴权 —— 探活方（k8s liveness/readiness、Docker healthcheck、负载均衡）
        手里没有凭据，要鉴权的话探针永远是 401，探测就失去意义。
      · 状态码分两档 —— Redis 是硬依赖（会话元数据全在里面），挂了返 503，
        让编排系统把流量摘走；MCP 和 MySQL 只是降级（MCP 连不上会退回纯本地
        检索，MySQL 连不上则记忆暂时退回进程内存），标 degraded 但仍是 200，
        不该因为一个可选组件把整个服务判死。
      · 探测函数自己吞异常并返回状态 —— 健康检查的职责是"报告"，把异常抛出去
        调用方只会拿到 500 堆栈，看不出到底哪个组件出问题。
    """
    checks = {
        "redis": await asyncio.to_thread(redis_status),  # ping 是阻塞网络 IO，别占事件循环
        "session_dir": (
            {"status": "ok", "path": str(SESSIONS_DIR)}
            if SESSIONS_DIR.exists()
            else {"status": "error", "detail": f"目录不存在：{SESSIONS_DIR}"}
        ),
        "mcp": mcp_status(),          # 只读全局状态，不主动拉起子进程
        # 注意这里是直接 await，没裹 asyncio.to_thread：mysql_status 本身就是
        # 异步函数（走 aiomysql），阻塞 IO 由驱动自己在事件循环里等。
        # 包一层 to_thread 反而会把 coroutine 对象丢进线程池，拿不到结果。
        "mysql": await mysql_status(),
        # 只读 lifespan 建好的那个全局，不在这里建连接 —— 健康检查不该有副作用。
        # 注意它和上面 mysql 探针回答的是两个不同的问题：mysql 说"此刻连得上库吗"，
        # 这里说"checkpointer 到底建起来没有"。库活着但建表失败（比如账号缺
        # CREATE 权限）时，前者 ok、后者 memory —— 正是这个组合最能说明问题。
        # 这两种状态都不参与下面的 overall 判定：MySQL 是软依赖，
        # 退化成进程内存只是记忆重启即丢，问答照跑，不该把整个服务判死。
        "checkpointer": (
            {"status": "ok", "backend": "mysql"}
            if checkpointer is not None
            else {
                "status": "memory",
                "detail": "MySQL 不可用或建表失败，对话记忆退化为进程内存（重启即清空）",
            }
        ),
    }

    overall = "ok"
    if checks["redis"]["status"] != "ok" or checks["session_dir"]["status"] != "ok":
        overall = "error"
    elif checks["mcp"]["status"] == "error" or checks["mysql"]["status"] == "error":
        # not_started 不算降级：MCP 是懒加载的，没人传过该领域的 PDF 就还没拉起来，
        # 这是正常初始态；只有明确报 error（上下文在但 session 是空的）才是真降级。
        # MySQL 同理：连不上只是记忆退化成进程内存，问答本身照跑，不该判死。
        overall = "degraded"

    response.status_code = 503 if overall == "error" else 200
    return {
        "status": overall,
        "checks": checks,
        # 会话数挂在外层图省事，前端/脚本一眼能看到；Redis 挂了就没有这个字段
        "active_sessions": checks["redis"].get("active_sessions"),
    }
