"""MySQL 连接配置、连接池与健康探测。

为什么单独开一个模块：后端、健康检查、LangGraph 的 checkpointer / store，都要用
同一份连接参数。散在各处迟早会出"健康检查连的是 A 库、真正写进去的是 B 库"这种
最难查的问题 —— 所以参数只在这里拼一次（mysql_connect_kwargs），谁用谁取。

三件事：给连接参数、建连接池、探活。建表不在这里做，归用它的 saver / store
自己 setup()（那是幂等的）。
"""

import os

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

# 为什么这里不给官方文档推荐的 DSN（saver.from_conn_string("mysql://user:pass@host/db")）：
# 那条路上有两个静默陷阱，都实测过，都不报错，只会在跑起来之后变成
# "密码明明是对的却 Access denied"和"字符集其实没设上"。
#
#  1. 密码里的 @ : / % 或空格会让 URL 解析跑偏。按文档调用方得自己 quote_plus
#     转义，但包内部是用 urlparse(...).password 取回来的，**不做反转义** ——
#     送进驱动的仍是 "p%40ss" 而不是 "p@ss"，认证直接失败。密码里只要有一个 @
#     就中招。（顺带：quote_plus 把空格编成 "+"，那是查询串的约定，在 userinfo
#     段是错的。）
#
#  2. 查询串被整个丢弃。parse_conn_string 走 urlparse + parse_qsl，只取
#     host / user / password / db / port / unix_socket 六个字段，尾部挂的
#     ?charset=utf8mb4 不报错也不生效。比报错更糟：你会以为设上了，实际连接用的是
#     服务端默认字符集，中文写进去迟早出问题 —— 而"迟早"意味着测试阶段大概率
#     看不出来。
#
# 所以这里不拼 DSN，改成把连接参数直接交给驱动：aiomysql 的 charset 是真实参数，
# 不经任何 URL 编解码，两个坑一起绕开。代价是绕开了官方入口 —— 连接池要传给
# saver 时只能走 AIOMySQLSaver(conn=pool) 这个鸭子类型接住的非文档化路径
# （细节和风险见 backend.py 的 _init_checkpointer）。
#
# 字符集仍然要在库层面钉死（建库时 CHARACTER SET utf8mb4）：连接参数管的是
# 这条连接，存进去之后怎么落地是服务端说了算。docker-compose 的 mysql 服务与
# 本机建库语句都显式指定了。


def mysql_connect_kwargs(**overrides):
    """一份连接参数，健康探测和连接池共用。

    共用是为了堵死"探测连的是 A 库、真正写进去的是 B 库"—— 两边同源，要错
    一起错，不会出现探测说 ok 而写入失败这种最难查的状态。
    overrides 给调用方补各自的差异（探测要短超时、池要长超时之类）。
    """
    kwargs = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "db": MYSQL_DATABASE,
        # 字符集只在这条路径上真正生效（见上面注释里的坑 2）
        "charset": "utf8mb4",
    }
    kwargs.update(overrides)
    return kwargs


async def create_mysql_pool(minsize=1, maxsize=5):
    """建一个 aiomysql 连接池。

    saver 和 store 各建一个（同一函数、不同尺寸），不共用。池本身的配置两边
    完全一样，没必要拆成两个函数；尺寸分开的理由见下面 maxsize 那段。

    为什么是池，而不是官方文档里的单连接（from_conn_string）：

      单连接那条路是 `async with aiomysql.connect(...)` 攥住一条连接用到底，
      中间既不 ping 也不重连。MySQL 默认 wait_timeout=28800（8 小时）会把这
      条空闲连接从服务端单方面掐掉，客户端毫无察觉 —— 表现是服务平稳跑满 8 小时
      后，所有 checkpoint 写入开始持续失败，且不会自愈，直到进程重启才恢复。
      这种"上线当天没事、第二天早上开始全挂"的故障最难排查。

      池的 pool_recycle 会在取用连接时按年龄主动丢弃重建（实现见 aiomysql
      Pool._acquire），这个坑从机制上就消失了。而且 saver 每次读写都是
      `async with get_connection(...)` 取还一次连接 —— get_connection 对池走的是
      pool.acquire()/release（见 langgraph .../mysql/_ainternal.py），所以年龄检查
      每次读写都会跑到，不存在"连接一直没人取、年龄检查不到"的死角。

    maxsize 为什么取这个量级：saver 和 store 内部**各自**都有一把 asyncio.Lock
    把所有读写串起来（checkpoint/mysql/aio_base.py:42、store/mysql/aio_base.py:49），
    所以每个池同一时刻真正被占用的连接都很少。数字给的是余量，不是并发需求 ——
    minsize=1 让空闲时只维持一条连接，maxsize 只是天花板，不预先开。
    健康探针不从这里取连接（mysql_status 每次现开现关），所以没给它留额度。

    两个池不共用是刻意的：saver 挂在图执行的每一步关键路径上，独立池能让 store
    的批量读写抢不走它的连接；反过来 store 建表失败也不会连累对话记忆。
    """
    return await aiomysql.create_pool(
        **mysql_connect_kwargs(),
        autocommit=True,      # checkpoint 是单语句写入，用不上事务；也让 setup() 的建表立刻生效
        minsize=minsize,
        maxsize=maxsize,
        pool_recycle=3600,    # 1 小时回收一次，离 8 小时的 wait_timeout 留足余量
        connect_timeout=3,    # 库挂了要快速失败，别把应用启动卡住
    )


async def mysql_status():
    """探一次 MySQL，返回健康检查用的状态块。任何情况下都不抛异常。

    每次现开一条连接、探完就关。为什么不复用一条长期连接：健康检查要回答的是
    "此刻连得上吗"，手里攥着一个早就断掉的连接对象，只会把 error 误报成 ok。
    """
    conn = None
    try:
        conn = await aiomysql.connect(
            # 和连接池同一份参数，只把超时改短：探针 3 秒还没通就该报 error，
            # 让它一直挂着比报错更糟 —— 健康检查自己有超时，会直接判整个探测失败。
            **mysql_connect_kwargs(connect_timeout=3),
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
