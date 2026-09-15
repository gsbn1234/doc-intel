# 后端镜像：FastAPI + LangGraph Agent-RAG
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装 torch。本机是 CPU 版（2.13.0+cpu），但 CPU 专用 wheel 国内无镜像
# （官方源被 GFW 限速到 60KB/s）。清华 PyPI 的 torch 2.13.0（502MB，打包 CUDA 运行时）
# 无 GPU 时自动跑 CPU，功能完全等价，清华源 3.6MB/s 约 2 分钟下完。
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.13.0

# 其余依赖全走清华源；torch 已装好，pip 这里直接跳过不会重装
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 业务代码。.env 故意不 COPY —— 里面是 API Key，只允许运行期由 compose env_file 注入
COPY multi_agent/ ./multi_agent/
COPY streamlit_1/ ./streamlit_1/
COPY mcp_tools/ ./mcp_tools/
COPY main.py ./main.py

EXPOSE 8000

# 多 worker（--workers N）：uvicorn 起 N 个独立进程，各自一套事件循环和 lifespan。
# 进程内状态（store、连接池、会话 LRU、MCP 子进程）随之变成每进程一份；
# 跨进程要共享的那几样（Redis 会话元数据、MySQL 记忆、faiss_db 磁盘文件）
# 本来就是外部服务/共享卷，多进程下照常只有一份 —— 这正是 Step 1-3 换掉
# 内存版 checkpointer/store 的回报。
#
# 默认 2 的由来：实测一个 worker 把 embedding + reranker 载满约 800MB（框架 439MB
# + 两个模型 360MB），两个约 1.6GB。取 2 是为了"多进程真的跑通"这件事有个最小
# 验证，不是因为再多就跑不动。改数量不用重建镜像（backend 镜像含 torch，重建很贵），
# compose 里加一行即可：  environment: [UVICORN_WORKERS=4]
#
# 为什么是 sh -c + exec 而不是直接 exec 形式：
#   ① exec 形式（JSON 数组）不做 shell 变量展开，要用环境变量覆盖就得过 shell；
#   ② 但 shell 形式若只写 `sh -c "uvicorn ..."`，PID 1 就成了 sh 而不是 uvicorn，
#      docker stop 的 SIGTERM 停在 sh 上不往下传，容器只能干等 10 秒被 SIGKILL ——
#      lifespan 里关连接池、关 MCP 子进程全来不及跑，MySQL 侧会积下十几个半开连接
#      直到 wait_timeout 才回收。exec 让 sh 用 uvicorn 替换掉自己，信号直达。
#
# 另外注意：uvicorn 的 --workers 走 multiprocessing spawn（Linux 上也是 spawn，
# 不是 fork），每个 worker 是把模块重新 import 一遍，所以上面那些进程内全局
# 在每个 worker 里都是各自新建的实例，不会互相干扰。
#
# 0.0.0.0：容器端口要对宿主机/别的容器可见，不能绑 127.0.0.1
CMD ["sh", "-c", "exec uvicorn streamlit_1.backend:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
