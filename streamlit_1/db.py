"""MySQL 连接配置与健康探测。

为什么单独开一个模块：后端、健康检查、以后可能出现的迁移脚本，都要用同一份连接
参数。散在各处迟早会出"健康检查连的是 A 库、真正写进去的是 B 库"这种最难查的问题。

这里只做两件事：拼 DSN、探活。不建连接池、不建表 —— 连接的生命周期归用它的
LangGraph saver / store 自己管（各自 from_conn_string）。
"""

import os
from urllib.parse import quote_plus

import aiomysql
from dotenv import load_dotenv

# 自己加载 .env，不去依赖"调用方碰巧先导入过 multi_agent.config"这个巧合。
# 现状是 backend.py 恰好先 import 了 multi_agent（那边 config.py 调了 load_dotenv），
# 所以顺序上没问题；但独立脚本（例如 Step 2 建表用的 python -m xxx）直接跑时
# 不会加载 .env，会静默退回 root + 空密码，报 Access denied 而很难联想到
# 真正原因是 dotenv 没加载。自己加载一次最省心。
#
# load_dotenv 幂等，且默认 override=False：
#   · 重复调用无副作用，不会把已设好的变量覆盖掉；
#   · 容器里 .env 不在镜像中（见 .dockerignore），它会静默 no-op，
#     配置继续来自 compose 注入的环境变量（MYSQL_HOST=mysql 等），两边不打架。
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "doc_app")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "doc_intel")

# 给 langgraph-checkpoint-mysql / -store 用的 DSN。两个坑，都踩过：
#
#  1. 用户名/密码必须 quote_plus。密码里只要出现 @ : / 之一，不转义就会把 URL
#     解析带偏，症状是"密码明明是对的却连不上"，很难往这上面想。
#
#  2. 千万别往查询串里加 ?charset=utf8mb4。这个包的 parse_conn_string 走的是
#     urlparse + parse_qsl，只取 host / user / password / db / port / unix_socket
#     六个字段，查询串被**静默丢弃** —— 比报错更糟，因为你会以为字符集设上了。
#     字符集只能在库层面钉死（建库时 CHARACTER SET utf8mb4），所以
#     docker-compose 的 mysql 服务与本机建库语句都显式指定了。
#
# 反过来说：我们自己用 aiomysql.connect 开连接时，charset 是真实参数、能生效
# （见下面 mysql_status）。被丢弃的只是 DSN 那条路径。
MYSQL_DSN = (
    f"mysql://{quote_plus(MYSQL_USER)}:{quote_plus(MYSQL_PASSWORD)}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)


async def mysql_status():
    """探一次 MySQL，返回健康检查用的状态块。任何情况下都不抛异常。

    每次现开一条连接、探完就关。为什么不复用一条长期连接：健康检查要回答的是
    "此刻连得上吗"，手里攥着一个早就断掉的连接对象，只会把 error 误报成 ok。
    """
    conn = None
    try:
        conn = await aiomysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            db=MYSQL_DATABASE,
            charset="utf8mb4",      # 这条路径能真正设上（见上面注释里的坑 2）
            connect_timeout=3,      # 库挂了要立刻返回，探针不能被拖死
        )
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
    except Exception as e:
        # 探测函数的职责是"报告是谁挂了"。抛出去调用方只会拿到一个 500 堆栈，
        # 反而看不出出问题的是哪个组件。
        return {"status": "error", "detail": f"{type(e).__name__}: {e}"}
    finally:
        if conn is not None:
            # 注意 close() 是同步方法（aiomysql 里 close() 不能 await，
            # 要 await 的是 ensure_closed()）。
            conn.close()
    return {"status": "ok", "database": MYSQL_DATABASE}
