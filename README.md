# doc-intel — 资料研究员：会自己查证的多 Agent 知识库问答系统

> **一句话定位**：不是又一个"上传 PDF 再聊两句"的问答 demo——是给你的资料库配一个
> **会自己查证的研究员**。它把每一次回答都当"研究任务"跑完整流程：
> **改写问题 → 判断查本地还是联网 → 多路检索（本地库 / Tavily 联网 / GitHub·arXiv·HuggingFace 专业工具）→ 成文 → 按"忠实 / 完整 / 相关"三规则审稿，不合格带修改意见打回重写**（最多两轮）。

**解决谁的什么问题**：学 AI 工程的人"收藏了几百份资料，提问时却答不出好答案"。
通用 RAG 只能复述资料里已有的话；而最容易答错、也最需要把关的恰恰是这三类问题——

1. **要跨文档查证的**：答案散在几份资料里 → Researcher 多路检索后再整合，而不是只翻到一段就答；
2. **要查最新信息的**：模型版本、行业动向，本地库收录不了 → router 判断走联网，而不是硬答"未找到"；
3. **要判断可信度的**：网上说法互相矛盾 → Reviewer 按"忠实于资料"审，防止把单一来源当全局结论。

**凭什么是"能用的"，不是 demo**（证据都在下文，可复现）：

- **有数字**：三档基线评测（同裁判同口径）完整链路 Context Recall 0.48→0.66、
  Answer Relevancy 0.50→0.74，且只有它能答联网时效题 → 见「评测与基线对比」。
- **能回归**：改一条检索策略就能一键重测（`python eval_baseline.py`），pytest + CI 全绿。
- **不装完美**：评测章节如实标注"单次裁判有波动，看方向不看绝对值"——知道评估的边界，
  不拿一个数字当真理。

**技术底座**：LangGraph 多 Agent 编排 · Parent Document 混合检索（FAISS + BM25）·
CrossEncoder 精排 · HyDE 假想答案 · MCP 工具 · FastAPI(SSE) + Streamlit · DeepSeek。
细节见「核心特性」与「架构决策记录（ADR）」。

> 📸 **演示位**：放一张"提问 → Researcher 调工具 → Reviewer 打回修订 → 出终稿"的截图或 GIF
> （本地相对路径引用），30 秒看懂它在干什么。

---

## 核心特性

- **多 Agent 协作**：Supervisor（调度）→ Researcher（本地库 + 联网搜索）→ Writer（成文）→ Reviewer（质量审核，不合格自动打回重搜），各节点各司其职，上下文互不污染
- **混合检索**：child 小块（200 字符）上做 FAISS + BM25 混合，映射回 parent 大块（800 字符），搜得准且上下文完整
- **检索增强**：Query Rewrite、Multi-Query、HyDE、CrossEncoder Rerank、Context Compression 全链路
- **长期记忆**：LangGraph Store 记住用户偏好（"记住…"）。对话 checkpoint 与长期偏好都持久化到 **MySQL**（`AIOMySQLSaver` + `AIOMySQLStore`），后端重启不丢
- **MCP 扩展**：Researcher 可调用自定义 MCP Server 工具
- **流式输出**：SSE（Server-Sent Events）实时推送搜索状态、工具调用和最终答案
- **可评估**：LLM-as-Judge 三维度（Context Recall / Faithfulness / Answer Relevancy）评测脚本

---

## 系统架构

```
                         ┌─────────────┐
                         │   用户提问   │
                         └──────┬──────┘
                                ▼
                        ┌───────────────┐
                        │ rewrite_query │  问题改写 + 写入长期记忆
                        └──────┬────────┘
                               ▼
                         ┌────────────┐
                    ┌───▶│ supervisor │◀───┐
                    │    └─────┬──────┘    │
                    │          │           │
               researcher      │       writer
                    │          │           │
   ┌────────────────┴──┐       │     ┌─────┴──────┐
   │ researcher_agent  │◀──────┘     │ writer_agent│ 生成最终回答
   └────────┬──────────┘             └──────┬─────┘
            │ 工具调用                        │
            ▼                                ▼
   ┌────────────────┐                ┌──────────────┐
   │  local_search  │  FAISS+BM25    │ reviewer_agent│ 质量审核
   │ internet_search│  Tavily 联网    │  忠实/完整/相关│
   │  MCP tools     │  自定义 MCP 工具 └──┬────┬──────┘
   └────────────────┘             PASS │    │ REVISE(未超上限)
                                      ▼    ▼
                                    END  researcher_agent（带修改意见重搜）
```

前端 Streamlit(8501) ↔ HTTP/SSE ↔ FastAPI(8000) ↔ LangGraph Agent-RAG

---

## 架构决策记录（ADR）

每个决策先交代"备选方案是什么、为什么没选"，面试被追问时按这个思路答。

### ADR-1：为什么用 LangGraph，而不是 CrewAI / AutoGen / 手写状态机

**需求驱动**：本系统核心是"Reviewer 审核不过 → 打回 Researcher 带修改意见重搜"的**循环**，
以及**多轮对话持久化**（checkpoint）。需要的是可控制的图结构，不是自由的对话流水线。

- **LangGraph**：`StateGraph` + 条件边 + checkpoint 天然表达"循环直到达标"，能精确控制
  每轮的工具调用和审核回路；`stream_mode` 还支持 `values`（节点状态）+ `messages`（token）双路流式。
- **CrewAI**：偏任务流水线（Role/Task 编排），对复杂循环、checkpoint 记忆的控制较弱。
- **AutoGen**：对话式多智能体，适合"辩论/讨论"型协作；对"检索→写作→审核→打回"这种确定性流程反而不可控。
- **手写状态机**：要自己实现 checkpoint、并发、流式，工程量不划算。

### ADR-2：为什么工具走 MCP 协议，而不是直接把工具 `bind_tools` 给 Researcher

- **隔离性**：MCP server 是独立 stdio 子进程，工具崩溃不会拖垮 Agent 主进程。
- **标准协议**：MCP 是行业标准（Anthropic/OpenAI 均支持），工具可跨语言、跨框架复用，
  不绑死在 LangChain 生态里。
- **领域化白名单**：`mcp_tools/registry.py` 按领域（ai_learning/general/...）过滤要挂载的工具，
  新增领域只加"server 注册 + registry 映射"两处，不改接入链路。
- 代价：stdio 进程间通信多一次 IPC 开销；为复用子进程做了全局客户端 + 锁的单例管理。

### ADR-3：为什么父子两粒度切分（child 搜 / parent 返）

- 小 chunk（200 字符）检索**精度高**，但上下文碎片化；大 chunk（800 字符）上下文**完整**，但召回噪声大。
- 折中：在 child 上建向量 + BM25 索引做精检索，命中后映射回所属 parent 整块喂给 LLM——
  既搜得准，又不丢失上下文。

### ADR-4：为什么前后端分离 + SSE 流式

- Streamlit 直接调 graph 会把 LLM 长时间阻塞在网页会话里，无法逐 token 展示过程。
- 拆成 FastAPI + SSE 后：检索状态、工具调用、Writer token 全部实时推送，前端做打字机效果；
  后端可独立水平扩展，任意前端（网页/移动/CLI）复用同一套 API。

### ADR-5：为什么会话用三层存储（内存 LRU + Redis + 磁盘）

- 最贵的操作是"重载 embedding 模型 + 重建图"（几秒级、烧资源）——内存 LRU 缓存（上限 20 个）
  命中热会话时开销降到零。
- Redis 存会话**元数据**（多进程 / 后端重启共享）；磁盘存**重物**（向量库 / 切块 / bm25 pickle）。
- 冷启动路径：Redis 命中元数据 → 从磁盘重建 → 回填 LRU。重启不丢会话。

### ADR-6：为什么开多 worker（`--workers 2`），以及随之而来的三处"每进程一份"

单进程时所有进程内状态天然只有一份，多开之后必须挨个确认"这东西该不该共享"。盘完分两类：

**该共享的 —— 本来就是外部服务或共享卷，多进程下自动只有一份**

| 状态 | 落在哪 |
|------|--------|
| 会话元数据 | Redis |
| 向量库 / chunks / parent_docs | `faiss_db/` 磁盘（compose 里是共享卷） |
| 对话记忆（checkpoint） | MySQL |
| 长期偏好（store） | MySQL |

这四样**没有一个是进程内存**，所以不会出现"这个 worker 记得、那个不记得"。这正是前几步把
`MemorySaver` / `InMemoryStore` 换成 `AIOMySQLSaver` / `AIOMySQLStore` 的回报 ——
当时换的理由是"后端重启不丢"，顺带把多进程的前提也备好了。反过来说：**如果没做那两步，
`--workers 2` 是个纯粹的 bug 制造机**（同一段对话在两个进程里各记一半）。

> ⚠️ 但这一段推论当时**漏了一问**，全栈 compose 验证时真的翻了车 —— 四样状态确实都在
> MySQL，可 `AIOMySQLSaver` 的**初始化**失败了，而降级目标恰恰是进程内存。详见本节末尾的
> 「补充」。

**不该共享的 —— 每进程一份，也不需要改**

- **连接池**：每个 worker 各建 saver + store 两条，互不干扰。
- **会话 LRU**：命中率减半。同一个会话被分到另一个 worker 就走"从磁盘重建"的冷路径 ——
  结果正确，只是慢一次。这是**已知取舍不是缺陷**：要让重建好的图跨进程共享，得引入
  序列化或外部缓存，收益配不上复杂度。
- **MCP 子进程**：懒加载，哪个 worker 先接到上传哪个拉起，每个 worker 最多一个。

**实测验证**（`uvicorn --workers 2` 本地真跑，不是 stub）：

- uvicorn 日志里两个**不同**的 `Started server process [pid]`，关停时两个都走到
  `Application shutdown complete`，关停后残留 `my_mcp_server` 子进程 **0** 个；
- 库里 `doc_app` 常驻连接 **4** 条（2 进程 × 2 池 × `minsize=1`），关停后归 0；
- 上传一次 PDF 后连打 40 次 `/api/health`，`checks.mcp` 出现 `not_started` 21 次 +
  `connected` 19 次 —— **同一个接口两种结果**，一次证明两件事：两个 worker 都在接请求
  （不是"起了两个只有一个干活"），且进程内状态确实各进程一份。

内存账：一个 worker 载满 embedding + reranker 实测约 **800MB**（框架 439MB + 两个模型
360MB），两个约 1.6GB。默认取 2 是"多进程真的跑通"的最小验证，不是因为再多跑不动 ——
改数量不用重建镜像（backend 镜像含 torch，重建很贵），compose 里加一行 `UVICORN_WORKERS=4` 即可。

#### 补充：多 worker 的坑不在"状态存哪"，在"谁来初始化"（全栈 compose 验证时补记）

上面那张表回答的是**状态存在哪**。它漏问了第二个问题：**N 个进程怎么收敛到同一份共享存储上**。
答案是四个字：各自跑一遍 `setup()`。于是开局就撞车 ——

```
两个 worker 同时启动，各自执行 AIOMySQLSaver.setup()：
  CREATE TABLE IF NOT EXISTS ...         ← 幂等，没事
  INSERT INTO checkpoint_migrations (0)  ← 不幂等！两个进程同时读到"版本表为空"，
                                           同时 INSERT，晚的那个撞主键
→ IntegrityError (1062, "Duplicate entry '0' for key 'checkpoint_migrations.PRIMARY'")
→ 被 _init_checkpointer 的兜底 except 接住 → 那个 worker 静默降级成进程内存
```

症状是**行为分裂**而不是报错：容器起来后连打 30 次 `/api/health`，**28 次**报
`checkpointer={"status":"ok","backend":"mysql"}`、**2 次**报 `{"status":"memory"}`。
落到那个 worker 上的请求，对话记忆重启就丢；而且每次重启丢的是哪个 worker 还是**随机的**。

本机从没暴露过 —— 单 worker 时这个交错不可能发生，它是 `--workers 2` **引进来的**。
更值得记的是：**这个 bug 正好推翻了上面那段"四样都不在进程内存，所以不会分裂"的推论。**
四样的确都在 MySQL，但初始化失败后，降级目标恰恰是进程内存。**"状态存在哪"和"状态实际
落在哪"是两回事** —— 中间隔着一条初始化路径。

修法是 `db.langgraph_setup_lock()`：用 MySQL 咨询锁（`GET_LOCK`）把 `setup()` 串起来。
两个设计点值得记：

- **锁连接独占，不从连接池里取。** 从池里取会死锁：N 个 worker 各占一条连接在 `GET_LOCK`
  上排队，池一满，拿到锁的那个也借不到连接去真正跑 `setup()`，只能等锁超时。独占一条
  临时连接，占用就与池容量、worker 数都无关。
- **`GET_LOCK` 是连接级的**，连接一断自动释放。所以进程被 SIGKILL 也不会留下永久锁 ——
  比自己建一张锁表稳妥（锁表方案遇到进程猝死就是个要人工介入的死锁）。

修后实测 30/30 一致。同一次验证还顺带查出另一处：writer 节点在 **async 函数里同步调**
`store.get()`，而 `AIOMySQLStore` 的同步 `get` 内部是
`run_coroutine_threadsafe(self.aget(...), store._loop).result()` —— 在事件循环线程上等一个
丢给同一个循环的协程，**真死锁**（langgraph 的 `_check_loop` 会拦下来抛 `InvalidStateError`，
否则就是挂住）。单测全走 `InMemoryStore`，同步调用完全正常，所以这条路一直没被跑到。改成
`await store.aget(...)` 即可。

**把这条推广一下**：多进程化要问两遍 —— ① 状态存在哪；② **谁负责把它初始化成可用的样子，
N 个进程之间怎么不打架**。第二问对"建表 / 迁移 / 注册 / 预热"这类**只该做一次**的动作都成立。

---

## 评测与基线对比

同一份 5 题测试集、同一个 LLM 裁判（Context Recall / Faithfulness / Answer Relevancy，0-1 分），
三档系统只换检索链路，打分口径完全一致：

| 档位 | Context Recall | Faithfulness | Answer Relevancy |
|------|---------------|--------------|------------------|
| 无检索直答（LLM 裸答，无上下文） | 0.0000 | 0.0000 | 1.0000 |
| 单路检索（仅向量 top-k） | 0.4800 | 0.8000 | 0.5000 |
| 完整系统（改写+路由+父子+HyDE+精排+压缩） | 0.6600 | 0.8000 | 0.7400 |

- **无检索直答** Recall/Faithfulness 双 0 → 证明"检索增强"的必要性；
- **完整系统** Recall / Answer Relevancy 明显高于单路检索（0.48→0.66、0.50→0.74），
  且只有它能答联网时效题 → 高级检索组件不是炫技（Faithfulness 单次持平，属裁判波动，见下注）；
- 联网题（DeepSeek-R1）只有完整系统能答：router 正确识别时效题 → 走 Tavily → 作答。
  这个缺陷是评估暴露的（原 router 把无关键词的时效题误判为 local），已通过改进 router 提示词修复。

> 局限（面试主动说明）：测试集为自建 5 题小集；裁判为 LLM-as-Judge，单次运行有随机波动，
> 看方向不看绝对值。逐题明细见 `docs/eval_baseline_result.md`。复现：`python eval_baseline.py`。

---

## 性能与成本

> 实测于 2026-08-29，模型 deepseek-v4-flash（DeepSeek 开放平台）。
> 耗时 = 答案阶段墙钟时间；token = 答案阶段 LLM 输入/输出（裁判打分只计入总成本）。
> 一次运行即可复现：`python eval_baseline.py`。

| 档位 | 平均耗时/题 | 平均输入 token | 平均输出 token |
|------|-----------|---------------|---------------|
| 无检索直答 | 3.1s | 223 | 358 |
| 单路检索 | 1.8s | 1,006 | 230 |
| 完整系统 | 35.4s | 3,970 | 2,168 |

- **一次完整三档基线评估（3 档 × 5 题 + 15 次裁判）≈ ¥0.12**：
  78.7k 输入 + 22.1k 输出 token，按官网价折算（输入 ¥1/百万·缓存未命中、输出 ¥2/百万）。
  DeepSeek 已于 2026-08 公告将上调价格，成本需按当时官网价复核。
- **完整系统单题比单路检索慢约 20×**：这是多轮 LLM 调用的代价
  （改写 + 路由 + HyDE + 多查询 + 生成；走本地检索时这 5 个各调一次，联网题少一次多查询）；
  换来的回报是 Recall 0.48→0.66，且只有它能答联网时效题。本地检索 + 精排本身 <1s，35s 大头在 LLM 生成。
- 生产环境压延迟的方向：DeepSeek 自动上下文缓存、子查询并行、更小生成模型。

> **口径说明**：本表三档走的都是 `eval_baseline.py` 的检索流水线 —— 档位 3 =
> `build_rag_graph(use_hyde=True)`（`multi_agent/router_graph.py`），**不含**多 Agent 的
> Researcher 工具循环与 Reviewer 审核打回。多 Agent 图（`multi_agent_graph.py`）由 `evaluate.py`
> 单独评，延迟更高，不在本表口径内。

---

## 技术栈

| 分类 | 技术 |
|------|------|
| 编排 | LangGraph、LangChain |
| 模型 | DeepSeek API（deepseek-v4-flash）、BAAI/bge-small-zh-v1.5（Embedding）、BAAI/bge-reranker-base（Rerank） |
| 检索 | FAISS 向量库、rank_bm25、jieba 中文分词 |
| 联网 | Tavily Search API |
| 服务 | FastAPI + SSE、Streamlit、MCP（FastMCP） |
| 记忆 | AIOMySQLSaver（MySQL，多轮对话 checkpoint）+ AIOMySQLStore（MySQL，长期偏好），各用一条独立连接池 |
| 部署 | Docker Compose（redis / mysql / backend / frontend 四服务）+ uvicorn 多 worker（默认 2，`UVICORN_WORKERS` 可调） |
| 可观测 | LangSmith 全链路追踪（LLM 调用 / Agent 步骤 / 工具轨迹） |
| 测试 | pytest（tests/ 目录，离线测试，不联网不烧 token） |

---

## 目录结构

```
doc-intel/
├── main.py                    # 脚本入口：加载 docs → 建索引 → 跑多 Agent 图
├── multi_agent/               # 核心库
│   ├── config.py              #   统一路径与配置（以项目根为基准）
│   ├── loader.py              #   PDF 文档加载
│   ├── parent_splitter.py     #   父子两粒度切分（child 搜 / parent 返回）
│   ├── embedding.py           #   bge 向量化
│   ├── vector_db.py           #   FAISS 建库 / 加载（路径统一走 config）
│   ├── bm25.py                #   BM25 关键词索引
│   ├── retriever.py           #   基础检索函数（vector / bm25）
│   ├── parent_retriever.py    #   父子混合检索（核心检索层）
│   ├── query_rewrite.py       #   查询改写
│   ├── multi_query.py         #   多角度查询生成
│   ├── hyde.py                #   HyDE 假想答案
│   ├── reranker.py            #   CrossEncoder 精排
│   ├── context_compressor.py  #   上下文压缩（逐句打分裁剪）
│   ├── web_retriever.py       #   Tavily 联网检索
│   ├── router_graph.py        #   Query Routing RAG 图（local/web）
│   ├── multi_agent_graph.py   #   多 Agent 协作图（Supervisor/Researcher/Writer/Reviewer）
│   └── llm.py                 #   DeepSeek 实例
├── streamlit_1/
│   ├── backend.py             # FastAPI 后端（上传 / 流式对话 / 健康检查）
│   ├── session_store.py       # 会话元数据（Redis）+ 索引目录管理
│   ├── db.py                  # MySQL 连接参数 / 连接池 / 健康探测
│   └── app.py                 # Streamlit 前端
├── mcp_tools/                 # 自定义 MCP Server（领域化工具注册）
│   ├── registry.py            #   领域 → 工具白名单（ai_learning/general/...）
│   ├── my_mcp_server.py       #   FastMCP stdio server（暴露全部领域工具）
│   └── tools/
│       └── ai_learning.py     #   AI 学习领域真实工具（GitHub / arXiv / HuggingFace）
├── docs/                      # 知识库 PDF（喂给 RAG 的原始文档）
├── evaluate.py                # LLM-as-Judge 三维度评估
├── eval_compare.py            # HyDE 开关对比评估
├── eval_dataset.py            # 评估测试集
├── test_backend.py            # FastAPI 链路冒烟测试（手动脚本，需真后端）
├── pytest.ini                 # pytest 配置（testpaths=tests，只收 tests/）
├── tests/                     # pytest 单元测试（离线，不联网）
│   ├── conftest.py            #   把项目根加入 sys.path
│   ├── test_config.py         #   路径 / 环境变量体检
│   ├── test_retriever.py      #   BM25 / 向量检索 / 父子映射
│   ├── test_auth_health.py    #   鉴权 / 健康检查 / lifespan 建池与降级
│   ├── test_async_offload.py  #   阻塞操作有没有占住事件循环
│   ├── test_chat_user_id.py   #   会话与 user_id 的隔离
│   ├── test_eval_judge.py     #   评估裁判的解析逻辑
│   └── test_multi_agent_graph.py # 图结构 + Reviewer 行为 + 答案提取
├── langgraph.json             # langgraph-cli 配置
└── faiss_db/                  # 生成的向量索引（勿提交 git）
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt   # 或 uv sync
```

### 2. 配置环境变量

复制 `.env` 并填入你的 Key：

```ini
DEEPSEEK_API_KEY=你的DeepSeekKey
TAVILY_API_KEY=你的TavilyKey
LANGSMITH_API_KEY=可选
LANGSMITH_TRACING=false
BACKEND_API_KEY=可选，见下
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=doc_app
MYSQL_PASSWORD=
MYSQL_DATABASE=doc_intel
MYSQL_ROOT_PASSWORD=
```

**`BACKEND_API_KEY`（后端接口鉴权）**

CORS 只约束浏览器，`curl` / `requests` 直接打 `:8000` 是绕得过去的，所以接口另加一层 API Key：

- **留空** = 不校验（本地开发默认）。后端启动时会打一条 WARNING 提醒，仅限本机使用。
- **填了值** = 两个上传/对话接口都要求带上这个 Key，校验走 `secrets.compare_digest`（常量时间比较，防时序侧信道）。请求头两种写法都认：
  ```bash
  curl -H "Authorization: Bearer $BACKEND_API_KEY" ...
  curl -H "X-API-Key: $BACKEND_API_KEY" ...
  ```
  缺 Key 或 Key 错误统一返回 `401`。
- 前端 `streamlit_1/app.py` 读的是**同一个变量名**，`docker-compose.yml` 里已给 frontend 服务注入，本地跑时两个进程都要能看到这个变量（同一个 `.env` 即可）。

**`MYSQL_*`（对话记忆持久化）**

把 LangGraph 的对话记忆与长期记忆落到 MySQL。**不配也能跑**——连不上只在健康检查里报 `degraded`（HTTP 仍是 200），问答不受影响。

- 本机直跑时用上表默认值（`localhost:3306`）。首次需要自己建库和账号，字符集必须当场钉死：
  ```sql
  CREATE DATABASE doc_intel CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
  CREATE USER 'doc_app'@'localhost' IDENTIFIED BY '<MYSQL_PASSWORD 的值>';
  GRANT ALL PRIVILEGES ON doc_intel.* TO 'doc_app'@'localhost';
  ```
  不用 root 跑应用，也**不要**图省事授权到 `*.*`——`doc_app` 只需要 `doc_intel` 这一个库。
  为什么不能省字符集：LangGraph 的 DSN 解析（`parse_conn_string`）只取 `host/user/password/db/port/unix_socket`，DSN 里写 `?charset=utf8mb4` 会被**静默丢弃**——比报错更麻烦，因为你会以为设上了。同理，容器里由 `--character-set-server` 在服务端指定。
- `MYSQL_ROOT_PASSWORD` **只给 docker compose 用**（本机直跑不读它），且**必须与 `MYSQL_PASSWORD` 不同**：两者同源的话，应用凭据一泄露 root 也跟着泄露。它没有默认值，没配就让 compose 当场报错，而不是悄悄退化成一个更弱的配置。
- `docker compose` 起时，`MYSQL_HOST` 被 compose 覆盖成服务名 `mysql`，`.env` 里的值不生效。MySQL 容器**不对宿主机暴露 3306**（`docker compose exec mysql mysql -uroot -p` 进去调试）。
- 版本锁在 **MySQL 8.0**：`mysql:8.0`。MySQL ≥ 9.6 在生成列中移除了 `MD5`，而 saver 的 `checkpoint_ns_hash` 正是用 `MD5` 生成列，官方未提供迁移路径——不要随手升级镜像 tag。

`GET /api/health` **不需要鉴权**——探活方（Docker healthcheck、k8s 探针、负载均衡）手里没有凭据，要鉴权的话探针永远是 401。它逐个报告依赖：

```json
{
  "status": "ok",                       // ok | degraded | error
  "checks": {
    "redis":        {"status": "ok", "active_sessions": 3},
    "session_dir":  {"status": "ok", "path": "/app/faiss_db/sessions"},
    "mcp":          {"status": "not_started"},
    "mysql":        {"status": "ok", "database": "doc_intel"},
    "checkpointer": {"status": "ok", "backend": "mysql"},
    "store":        {"status": "ok", "backend": "mysql"}
  },
  "active_sessions": 3
}
```

- Redis 或索引目录出问题 → `status: "error"`，HTTP **503**（硬依赖挂了，该把流量摘走）。
- 只有 MCP 或 MySQL 出问题 → `status: "degraded"`，HTTP 仍是 **200**（MCP 会退回纯本地检索，MySQL 挂了记忆退回进程内存，都不该因此把整个服务判死）。`not_started` 是 MCP 懒加载的正常初始态，不算降级。
- `checkpointer` 和 `store` 回答的是**各自**的问题，不能互相代表：`mysql` 说"此刻连得上库吗"，这两个说"东西建起来没有"。库活着但表没建起来（比如账号缺 `CREATE` 权限）时就是 `mysql: ok` + 两个都报 `memory` —— 这个组合最能说明问题。两者各用一条独立连接池、各自独立降级，所以"一个成了另一个没成"是真实可能的，健康检查分开报就是为了不把这种半边坏掉的情况掩盖成"一切正常"。
- `session_dir` 报的是**绝对路径**（以项目根为基准，容器里是 `/app/faiss_db/sessions`，正好落在 compose 的 `./faiss_db:/app/faiss_db` 卷里）。以前它是相对 CWD 的 `faiss_db/sessions`，只在"启动目录恰好是项目根"时才落对地方；启动方式一变（换 `WORKDIR`、systemd、从上级目录 `python -m`）就会静默写到别处 —— 目录照建、读写正常，只是数据在卷外，容器一重建全丢。健康检查把路径打出来，就是为了这种情况能一眼看见。
- 开了多 worker 后要留意：`mcp` 是**进程内**状态，同一个接口不同请求可能返回不同结果（哪个 worker 接过上传，哪个就是 `connected`，其余是 `not_started`；两个都算正常）。`checkpointer` / `store` 不受影响 —— 它们后端是 MySQL，每个 worker 都连得上。

**接口文档：<http://127.0.0.1:8000/docs>**

后端启动后就有 Swagger UI，不用读源码就能试接口。三个接口的响应都用 Pydantic 响应模型声明过
（`UploadResponse` / `HealthResponse` / `ComponentCheck`），所以文档里的字段名、类型、取值集合
（`status` 只能是 `ok`/`degraded`/`error`）以及 `401`/`404`/`413`/`503` 各状态码的含义都是自动生成的，
改了代码文档跟着变。

`/api/chat-stream` 是唯一的例外：它是 SSE 长连接，响应体不是"一次成形的一个 JSON 对象"，
`response_model` 那套前提不成立，所以它的事件契约（6 种 `type`）写在 `responses` 里。

### 3. 启用 LangSmith 追踪（可选）

在 `.env` 中把 `LANGSMITH_TRACING` 改为 `true` 并填入真实 `LANGSMITH_API_KEY`
（[smith.langchain.com](https://smith.langchain.com) → Settings → API Keys，`lsv2_` 开头），
`LANGSMITH_PROJECT` 建议改成 `doc-intel`。不填 Key 或保持 `false`，程序静默关闭追踪，不影响任何功能。

开启后，每次跑图（`main.py` / 后端流式对话）都会自动上报：每一步 LLM 的输入输出、token 消耗、
Researcher 的工具调用、Reviewer 的审核结论。在 LangSmith 里可按 `run_name`（`multi_agent_rag` / `chat_stream`）筛选。

### 4. 运行方式（三选一）

**方式 A：脚本直接跑（最快）**
```bash
python main.py
```

**方式 B：Web 全栈（FastAPI + Streamlit）**

终端 1 —— 启动后端：
```bash
uvicorn streamlit_1.backend:app --port 8000 --reload     # 开发：单进程，改代码热重载
uvicorn streamlit_1.backend:app --port 8000 --workers 2  # 多进程：见 ADR-6
```
终端 2 —— 启动前端：
```bash
streamlit run streamlit_1/app.py
```
浏览器打开 `http://localhost:8501`，上传 PDF → 构建索引 → 提问。

> `--reload` 和 `--workers` 互斥，别同时给。多进程下每个 worker 各占约 800MB（模型各载一份），
> 按机器内存给数量；容器里由 `UVICORN_WORKERS` 控制，默认 2。

### 5. 跑评估

```bash
python evaluate.py            # 三维度评估
python eval_compare.py        # HyDE 开关对比
python eval_baseline.py       # 三档基线对比（无检索 vs 单路 vs 完整）
```

### 6. 跑单元测试（pytest）

```bash
# 首次先装测试依赖
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 跑全部测试（全部离线：假 LLM + 合成文档，不联网、不烧 token）
.venv\Scripts\python.exe -m pytest
```

测试覆盖：`config` 路径/环境变量、BM25 / 向量检索 / 父子映射、
多 Agent 图结构（节点/边线）、Reviewer 节点行为、答案提取。

> 注：`pytest.ini` 里 `testpaths = tests`，只收集 `tests/` 目录。
> 项目根的 `test_backend.py` 是手动冒烟脚本（需真后端 + 真 API），不会被 pytest 误收集。

---

## 常见问题

- **换目录跑就报错 / 找不到 docs**：所有路径统一在 `multi_agent/config.py` 中按项目根计算，请勿自行硬编码相对路径。
- **faiss_db 索引过期**：`docs/` 内容更新后需删除 `faiss_db/` 重新建索引（目前索引按"存在即复用"策略，暂未自动校验文档变更）。
- **记忆重启丢失**：对话记忆（多轮对话 checkpoint，`AIOMySQLSaver`）和长期记忆（用户偏好，`AIOMySQLStore`）都存在 **MySQL** 里，后端重启不丢。两者的建表都在启动时由各自的 `setup()` 自动完成，各用一条独立连接池（saver 挂在图执行的每一步关键路径上，独立池能让 store 的批量读写抢不走它的连接）。若启动时连不上 MySQL 或建表失败，对应那一项会**降级**为进程内存（重启即清空）并在日志里留 warning，`/api/health` 里如实报 `{"status": "memory"}` 而不是笼统的 ok —— 库活着但账号缺 `CREATE` 权限就正好是这个组合。注意长期记忆降级的是**内存版 store 而不是空值**：图里的 `rewrite_query` 节点会真的调它读写偏好，给空值会直接让问答报错，比"偏好重启后丢了"严重得多。

---

## 备份说明

清理死代码时删除的旧版本模块已打包至
`D:\python2\lc_course_已删除代码备份_20260822.zip`（含 `legacy/`、`LangChainRAG/` 及误建的残留目录），如需找回可解压恢复。
