"""应用级接线的测试：接口鉴权、健康检查的依赖探测、生命周期钩子。

背景（三件事）：
  1. 没有鉴权。CORS 只锁来源，它约束的是浏览器；curl / requests 直接打 :8000
     完全绕开，谁都能上传 PDF、跑 Agent、烧 token。加一层 API Key。
  2. /api/health 不查依赖。原来固定返回 {"status": "ok"}，Redis 挂了照样说 ok，
     探活方被蒙在鼓里。合格的健康检查要逐个报告组件状态。
  3. 关停钩子用的是已弃用的 @app.on_event("shutdown")，换成 lifespan。

这里锁死的行为：
  · 没配 Key → 放行（本地开发默认）；配了 Key → 不带/带错都是 401
  · Bearer 和 X-API-Key 两种头都认
  · /api/health 不需要鉴权（探针没凭据，要鉴权的话探针永远 401）
  · Redis / 索引目录挂了 → 503；MCP 或 MySQL 出问题 → degraded 但仍是 200
    （软依赖挂了不代表服务不能干活，判 503 反而会让编排系统误摘流量）
  · 关停时真的会去 close_mcp（这类钩子漏挂平时没症状，只有退出时才看得出来）
  · 对话记忆的 MySQL checkpointer：建起来了要收池、要复位全局；建不起来只降级
    （不抛异常、不泄漏池），健康检查里如实报 memory 而不是笼统的 ok
  · 长期记忆的 MySQL store：同上，但降级目标是**能用的内存 store**而不是 None，
    且它走自己独立的连接池（不和 checkpointer 共用）

不碰网络、不碰真 Redis：外部依赖全换成桩。
"""
import asyncio
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

import streamlit_1.backend as backend


# ========== 一、鉴权 ==========

class _FakeGraph:
    """只回一个终止态状态块，让请求能走到流式响应结束。"""

    async def astream(self, state, config=None, stream_mode=None):
        yield ("values", {"next_agent": "finish"})


async def _fake_get_session(session_id, checkpointer=None, store=None):
    return {"graph": _FakeGraph()}


def _chat(headers):
    """打一次对话接口，返回响应。用桩会话，绕开真实的索引重建。"""
    with patch.object(backend, "get_session", _fake_get_session):
        return TestClient(backend.app).post(
            "/api/chat-stream",
            json={"session_id": "s1", "question": "你好", "user_id": "alice"},
            headers=headers,
        )


def test_no_key_configured_allows_anonymous(monkeypatch):
    """没配 BACKEND_API_KEY = 本地开发模式，不校验，请求照常放行。"""
    monkeypatch.setattr(backend, "API_KEY", "")
    assert _chat({}).status_code == 200


def test_key_configured_rejects_missing_header(monkeypatch):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    resp = _chat({})
    assert resp.status_code == 401, resp.text
    # 带上 WWW-Authenticate，符合 HTTP 认证失败的约定
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    assert "API Key" in resp.json()["detail"]


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer wrong-key"},          # 值不对
        {"Authorization": "Bearer s3cret-extra"},       # 只是前缀对，也要拒
        {"Authorization": "Bearer "},                   # 空 token
        {"Authorization": "s3cret"},                    # 少了 Bearer 前缀 → 不是可识别的凭据
        {"Authorization": "Basic czNjcmV0"},            # 换了认证方案
        {"X-API-Key": "wrong-key"},
        {"X-API-Key": ""},
    ],
)
def test_key_configured_rejects_bad_credentials(monkeypatch, headers):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    resp = _chat(headers)
    assert resp.status_code == 401, f"{headers} 应被拒，实际 {resp.status_code}"


def test_bearer_header_accepted(monkeypatch):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    assert _chat({"Authorization": "Bearer s3cret"}).status_code == 200


def test_x_api_key_header_accepted(monkeypatch):
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    assert _chat({"X-API-Key": "s3cret"}).status_code == 200


def test_bearer_prefix_is_case_insensitive(monkeypatch):
    """HTTP 认证方案名大小写不敏感（RFC 7235），别只认小写。"""
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    assert _chat({"Authorization": "bearer s3cret"}).status_code == 200


def test_upload_route_is_also_protected(monkeypatch):
    """上传接口同样挂了依赖：不带 Key 时在校验文件之前就被挡下（401 而不是 422）。"""
    monkeypatch.setattr(backend, "API_KEY", "s3cret")
    resp = TestClient(backend.app).post("/api/upload-pdf")
    assert resp.status_code == 401, resp.text


# ========== 二、健康检查 ==========

def _health(redis=None, mcp=None, mysql=None, sessions_dir=None, api_key="s3cret"):
    """打一次健康检查，各依赖换成指定桩。

    mysql 默认给"健康"：绝大多数用例关心的是 redis / mcp，不该被 MySQL 带偏。
    """
    if mysql is None:
        mysql = {"status": "ok", "database": "doc_intel"}

    async def _mysql_stub():
        # 必须是 async：backend 里是 `await mysql_status()`，桩返回裸 dict 的话
        # 会炸在 "object dict can't be used in 'await' expression"。
        return mysql

    patches = [
        patch.object(backend, "API_KEY", api_key),
        patch.object(backend, "redis_status", lambda: redis),
        patch.object(backend, "mcp_status", lambda: mcp),
        patch.object(backend, "mysql_status", _mysql_stub),
    ]
    if sessions_dir is not None:
        patches.append(patch.object(backend, "SESSIONS_DIR", sessions_dir))
    for p in patches:
        p.start()
    try:
        return TestClient(backend.app).get("/api/health")
    finally:
        for p in patches:
            p.stop()


def test_health_needs_no_auth(monkeypatch, tmp_path):
    """配了 Key 也不拦健康检查——探针手里没有凭据。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 3},
        mcp={"status": "not_started"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200, resp.text


def test_health_ok_when_all_dependencies_up(tmp_path):
    resp = _health(
        redis={"status": "ok", "active_sessions": 3},
        mcp={"status": "connected"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["active_sessions"] == 3
    assert set(body["checks"]) == {
        "redis", "session_dir", "mcp", "mysql", "checkpointer", "store",
    }
    assert body["checks"]["session_dir"]["path"] == str(tmp_path)


def test_health_503_when_redis_down(tmp_path):
    """Redis 是硬依赖（会话元数据全在里面）→ 503，让编排系统摘流量。"""
    resp = _health(
        redis={"status": "error", "detail": "ConnectionError: refused"},
        mcp={"status": "connected"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["status"] == "error"
    assert "refused" in body["checks"]["redis"]["detail"]
    # 探不到会话数就不给这个字段，而不是编一个 0。
    # 用 .get()：响应模型开了 exclude_none，为 None 的字段整个不出现（不是 null）。
    assert body.get("active_sessions") is None


def test_health_503_when_session_dir_missing(tmp_path):
    """索引目录不见了，上传/重建全会失败，同样算硬依赖故障。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 0},
        mcp={"status": "connected"},
        sessions_dir=tmp_path / "不存在的目录",
    )
    assert resp.status_code == 503
    assert resp.json()["status"] == "error"


def test_health_degraded_but_200_when_mcp_broken(tmp_path):
    """MCP 只是工具降级（连不上会退回纯本地检索），不该把整个服务判死。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 1},
        mcp={"status": "error", "detail": "session 为空"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_health_mcp_not_started_is_not_degraded(tmp_path):
    """MCP 是懒加载：没人传过该领域的 PDF 就还没拉起来，这是正常初始态。"""
    resp = _health(
        redis={"status": "ok", "active_sessions": 0},
        mcp={"status": "not_started"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_degraded_but_200_when_mysql_broken(tmp_path):
    """MySQL 是软依赖：连不上只是记忆退回进程内存，问答本身照跑。

    这里特意断言 200 而不是 503 —— 判 503 会让编排系统把一个其实能正常
    服务的实例摘掉流量，比不判还糟。
    """
    resp = _health(
        redis={"status": "ok", "active_sessions": 1},
        mcp={"status": "connected"},
        mysql={"status": "error", "detail": "OperationalError: Can't connect to MySQL server"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "degraded"
    assert "Can't connect" in body["checks"]["mysql"]["detail"]
    # 软依赖挂了不影响会话数照常上报
    assert body["active_sessions"] == 1


def test_health_hard_dependency_wins_over_soft(tmp_path):
    """硬依赖（Redis）和软依赖（MySQL）同时挂 → 503 优先。

    保证两档判定不会互相覆盖：先判硬依赖，硬依赖不 ok 就直接 error。
    """
    resp = _health(
        redis={"status": "error", "detail": "ConnectionError: refused"},
        mcp={"status": "error", "detail": "session 为空"},
        mysql={"status": "error", "detail": "OperationalError: refused"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 503
    assert resp.json()["status"] == "error"


def test_health_redis_probe_failure_is_reported_not_raised():
    """探测函数不抛异常：健康检查的职责是"报告是谁挂了"，
    抛出去调用方只会拿到 500，看不出是哪个组件。"""
    with patch("streamlit_1.session_store.r") as fake_r:
        fake_r.ping.side_effect = ConnectionError("refused")
        status = backend.redis_status()
    assert status["status"] == "error"
    assert "ConnectionError" in status["detail"]


# ========== 三、应用生命周期（lifespan） ==========

def test_shutdown_closes_mcp():
    """关停时要真的去关 MCP 子进程。

    坑：直接 `TestClient(app)` 不会跑 lifespan，必须用 `with TestClient(app)`。
    退出 with 块就等于走了一遍关停，所以断言写在 with 外面。
    """
    called = []

    async def fake_close():
        called.append(True)

    # lifespan 里除了关 MCP，还会去建两个 MySQL 连接池（checkpointer 一个、
    # store 一个）。这里都替换掉：这个用例只管 MCP，不该因为跑测试的机器上没
    # MySQL 就红，也不该白等 3 秒连接超时。
    async def no_mysql():
        return None, None

    async def no_mysql_store():
        # 注意 store 的降级目标是个能用的内存 store，不是 None —— 图里的
        # rewrite_query 节点会真的调它读写用户偏好，给 None 会在请求里炸。
        return backend.InMemoryStore(), None

    with patch.object(backend, "close_mcp", fake_close), \
         patch.object(backend, "_init_checkpointer", no_mysql), \
         patch.object(backend, "_init_store", no_mysql_store):
        with TestClient(backend.app):
            pass
    assert called, "关停时没关 MCP 子进程——它会挂在事件循环清理上刷噪音错误"


def test_lifespan_is_wired_into_the_app():
    """关停逻辑得真的挂上去。这类钩子漏挂平时完全没症状，
    只有退出时才发现（而且报的是子进程相关的噪音错误，很难联想到钩子没挂）。"""
    assert backend.app.router.lifespan_context is backend.lifespan, (
        "app 没用上我们自己定义的 lifespan——关停钩子等于没挂"
    )
    assert not backend.app.router.on_shutdown, (
        "还有 @app.on_event 注册的钩子——已弃用的写法没清干净"
    )


# ========== 四、响应模型（response_model） ==========

def _openapi():
    return TestClient(backend.app).get("/openapi.json").json()


def _json_schema_ref(spec, path, method, status="200"):
    return (
        spec["paths"][path][method]["responses"][status]
        ["content"]["application/json"]["schema"]
    )


def test_upload_response_model_is_declared():
    """漏掉 response_model 时 /docs 里响应是空的，前端只能靠读源码猜字段名。
    这条断言读的是自动生成的 OpenAPI，等于在验证"文档里真的有这个契约"。"""
    spec = _openapi()
    assert "UploadResponse" in spec["components"]["schemas"]
    assert _json_schema_ref(spec, "/api/upload-pdf", "post")["$ref"].endswith(
        "/UploadResponse"
    )


def test_health_declares_503_in_openapi():
    """503 是 Redis 挂了的正常业务路径，调用方得能从文档知道要处理它。"""
    spec = _openapi()
    responses = spec["paths"]["/api/health"]["get"]["responses"]
    assert "200" in responses and "503" in responses


def test_chat_stream_documents_its_sse_events():
    """SSE 用不了 response_model，事件契约只能靠 responses 写进文档。
    否则 /docs 里这个接口的响应只有一行 text/event-stream，什么都没说。"""
    spec = _openapi()
    desc = spec["paths"]["/api/chat-stream"]["post"]["responses"]["200"]["description"]
    for event in ("status", "writer_start", "token", "tool_call", "done", "error"):
        assert event in desc, f"/docs 里没说明 {event} 事件"


def test_unknown_check_fields_are_not_stripped(tmp_path):
    """探测器以后多返回一个字段，不能被 response_model 静默吃掉。

    这是给 ComponentCheck 配 extra="allow" 的原因：response_model 默认会把
    模型里没声明的字段剥掉，而且不报错——数据在 API 层凭空消失最难查。
    """
    resp = _health(
        redis={"status": "ok", "active_sessions": 1, "latency_ms": 7},
        mcp={"status": "connected"},
        sessions_dir=tmp_path,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["checks"]["redis"]["latency_ms"] == 7, (
        "未声明的字段被剥掉了——extra='allow' 没生效"
    )


def test_health_omits_absent_fields_instead_of_nulling_them(tmp_path):
    """开了 exclude_none：没值的字段整个不出现，不铺一地 "detail": null。
    健康检查的输出经常是人 curl 一下直接看的，干净比完整重要。"""
    body = _health(
        redis={"status": "ok", "active_sessions": 2},
        mcp={"status": "not_started"},
        sessions_dir=tmp_path,
    ).json()
    assert body["checks"]["redis"] == {"status": "ok", "active_sessions": 2}
    assert "detail" not in body["checks"]["redis"]
    assert "path" not in body["checks"]["mcp"]


# ========== 五、对话记忆（MySQL checkpointer 接线） ==========
#
# 这一节的背景是一个存在了很久、且完全没有症状的真实缺陷：原来的 SqliteSaver
# 是用 try/except ImportError 兜底的，而 langgraph-checkpoint-sqlite 从头到尾
# 没进过依赖 —— ImportError 被静默吞掉，checkpointer 永远是 None，对话记忆
# 一次都没落过盘（checkpoints.sqlite 一直是 0 字节）。
# 所以这里锁死的除了"能建起来"，还有"建不起来时要留痕、要收干净、且不拖垮启动"。

class _FakePool:
    """假连接池，只记 close / wait_closed 有没有被调到。"""

    def __init__(self):
        self.closed = False
        self.waited = False

    def close(self):
        self.closed = True

    async def wait_closed(self):
        self.waited = True


def _stub_init(saver, pool):
    """替掉 backend._init_checkpointer 的桩：直接给固定的 (saver, pool)。"""
    async def _init():
        return saver, pool

    return _init


def _stub_init_store(store, pool):
    """替掉 backend._init_store 的桩：直接给固定的 (store, pool)。"""
    async def _init():
        return store, pool

    return _init


# 跑 lifespan 的用例都得把这两个初始化都换掉：留着真的那个会去连测试机上的
# MySQL（连不上白等 3 秒超时，连得上则留下真实连接池和表）。下面用一对固定的
# 桩，顺便让"两个池是分开的"这件事可断言。
def _both_stubs(saver=None, store=None):
    """给 lifespan 用的一对桩 + 它们各自的假池，返回 (patches, ckpt_pool, store_pool)。"""
    ckpt_pool, store_pool = _FakePool(), _FakePool()
    saver = object() if saver is None else saver
    store = object() if store is None else store
    patches = [
        patch.object(backend, "_init_checkpointer", _stub_init(saver, ckpt_pool)),
        patch.object(backend, "_init_store", _stub_init_store(store, store_pool)),
    ]
    return patches, (saver, ckpt_pool, store, store_pool)


def test_lifespan_creates_checkpointer_and_closes_pool():
    """启动时把模块级 checkpointer 指向 MySQL saver；关停时收池并复位全局。

    复位那一步最容易漏：同一个进程里会反复进出 lifespan，不复位的话第二次
    拿到的就是指向已关闭池的悬空 saver（报错现场在 saver 里，看不出根因）。
    """
    patches, (saver, ckpt_pool, _store, _store_pool) = _both_stubs()

    for p in patches:
        p.start()
    try:
        with TestClient(backend.app):
            assert backend.checkpointer is saver, (
                "启动后全局 checkpointer 没指向 MySQL saver——"
                "多半是 lifespan 里漏了 `global checkpointer`，赋值落到局部变量上了"
            )
        assert ckpt_pool.closed, "关停时没关连接池"
        assert ckpt_pool.waited, "close() 只是标记关闭，必须 await wait_closed() 才算收完"
    finally:
        for p in patches:
            p.stop()

    assert backend.checkpointer is None, "关停后没复位，留下指向已关闭池的悬空引用"


def test_lifespan_creates_store_on_its_own_pool():
    """store 走自己那条连接池，和 checkpointer 的不是同一个。

    这是刻意的设计决定，不是实现细节：saver 挂在图执行的每一步关键路径上，
    两边共用一个池的话，store 的批量读写会抢走它的连接、把 checkpoint 写入拖住。
    所以这里断言的是"两个池对象不同 + 关停时两个都被收掉"，
    哪天有人图省事把两个池合并成一个，这条会红。
    """
    patches, (_saver, ckpt_pool, store, store_pool) = _both_stubs()

    for p in patches:
        p.start()
    try:
        with TestClient(backend.app):
            assert backend.store is store, (
                "启动后全局 store 没指向 MySQL store——"
                "多半是 lifespan 里漏了 `global store`，赋值落到局部变量上了"
            )
        assert store_pool is not ckpt_pool, "两个池不能是同一个对象"
        # 两个池各自独立关闭，漏掉任何一个都会把连接挂在 MySQL 侧
        assert ckpt_pool.closed and ckpt_pool.waited, "checkpointer 的池没收干净"
        assert store_pool.closed and store_pool.waited, "store 的池没收干净"
    finally:
        for p in patches:
            p.stop()

    assert isinstance(backend.store, backend.InMemoryStore), (
        "关停后 store 该复位成内存版（能用的降级态），而不是留着"
        "指向已关闭池的悬空引用 —— 图里的 rewrite_query 节点会真的调它"
    )


def test_init_checkpointer_degrades_when_mysql_unreachable():
    """MySQL 连不上只降级、不抛异常：软依赖挂了不该让整个服务起不来。"""
    async def boom():
        raise OSError("connection refused")

    with patch.object(backend, "create_mysql_pool", boom):
        saver, pool = asyncio.run(backend._init_checkpointer())

    assert (saver, pool) == (None, None), (
        "建池失败时该返回 (None, None)，让 build_multi_agent_graph 退回 MemorySaver"
    )


def test_init_checkpointer_closes_pool_when_setup_fails():
    """池建成了但建表失败：池必须收掉，不能泄漏。

    最容易出的岔子是把 setup() 的异常直接抛出去 —— 应用没起来，池却已经建好
    且没人收，连接就这么挂在 MySQL 上直到服务端超时。
    """
    pool = _FakePool()

    async def fake_create_pool():
        return pool

    class _BadSaver:
        def __init__(self, conn=None):
            assert conn is pool, "saver 必须拿到建好的那个池"

        async def setup(self):
            raise RuntimeError("CREATE TABLE 被拒（账号缺 CREATE 权限）")

    with patch.object(backend, "create_mysql_pool", fake_create_pool), \
         patch.object(backend, "AIOMySQLSaver", _BadSaver):
        saver, got_pool = asyncio.run(backend._init_checkpointer())

    assert (saver, got_pool) == (None, None)
    assert pool.closed and pool.waited, "建表失败后池没被收掉，连接泄漏了"


def test_health_tells_apart_mysql_probe_and_checkpointer(monkeypatch):
    """mysql 探针说 ok、checkpointer 说 memory —— 这个组合最能说明问题：
    库是活的，但表没建起来（比如账号缺 CREATE 权限）。两个字段必须各说各的。

    同时锁死：checkpointer 退化不参与 overall 判定。记忆退回进程内存只是
    重启即丢，问答照跑，不该把整个服务判成 degraded/error。
    """
    monkeypatch.setattr(backend, "checkpointer", None)
    body = _health(redis={"status": "ok", "active_sessions": 1},
                   mcp={"status": "connected"}).json()

    assert body["checks"]["mysql"]["status"] == "ok"
    assert body["checks"]["checkpointer"]["status"] == "memory"
    assert "detail" in body["checks"]["checkpointer"], "退化时要说明退到哪去了"
    assert body["status"] == "ok", "checkpointer 退化不该影响 overall"


def test_health_reports_mysql_backed_checkpointer(monkeypatch):
    """checkpointer 建起来时，状态要明确说是 MySQL 撑的，而不是笼统的 ok。"""
    monkeypatch.setattr(backend, "checkpointer", object())
    body = _health(redis={"status": "ok", "active_sessions": 1},
                   mcp={"status": "connected"}).json()

    assert body["checks"]["checkpointer"] == {"status": "ok", "backend": "mysql"}


# ========== 六、长期记忆（MySQL store 接线） ==========
#
# 和第五节同构，但有一处**关键差别**必须锁死：store 建不起来时不能降级成 None。
# checkpointer 给 None 是安全的（build_multi_agent_graph 内部会退回 MemorySaver），
# 但 store 给 None 会在请求里炸 —— 图里的 rewrite_query 节点会真的调它读写用户偏好。
# 所以 store 的降级目标是一个能用的 InMemoryStore。

def _real_mysql_store():
    """造一个真的 AIOMySQLStore 实例，只为让健康检查的 isinstance 判成 mysql。

    构造它需要一个运行中的事件循环（内部要 asyncio.get_running_loop()），
    但**不需要真连接**：conn 在构造期只是被存下来，不碰数据库。所以传个 object()
    就行，测试不必依赖测试机上有没有 MySQL。
    """
    async def _make():
        return backend.AIOMySQLStore(conn=object())

    return asyncio.run(_make())


def test_init_store_degrades_to_inmemory_when_mysql_unreachable():
    """MySQL 连不上时降级成内存 store，而不是 None。

    这条是本节的要点：传 None 会让"记住我的偏好"在用户提问时直接抛异常，
    比"偏好重启后丢了"严重得多 —— 前者是功能挂掉，后者只是降级。
    """
    async def boom(*args, **kwargs):      # 建池是带 min/maxsize 关键字调的
        raise OSError("connection refused")

    with patch.object(backend, "create_mysql_pool", boom):
        st, pool = asyncio.run(backend._init_store())

    assert isinstance(st, backend.InMemoryStore), (
        "store 降级目标必须是能用的内存 store；返回 None 会让 rewrite_query 节点炸"
    )
    assert pool is None, "池都没建成，不该返回池"


def test_init_store_closes_pool_when_setup_fails():
    """池建成了但建表失败：池必须收掉，不能泄漏。

    和 checkpointer 那边同理 —— 最容易出的岔子是把 setup() 的异常直接抛出去，
    应用没起来、池却已经建好且没人收。
    """
    pool = _FakePool()

    async def fake_create_pool(*args, **kwargs):
        return pool

    class _BadStore:
        def __init__(self, conn=None):
            assert conn is pool, "store 必须拿到建好的那个池"

        async def setup(self):
            raise RuntimeError("CREATE TABLE 被拒（账号缺 CREATE 权限）")

    with patch.object(backend, "create_mysql_pool", fake_create_pool), \
         patch.object(backend, "AIOMySQLStore", _BadStore):
        st, got_pool = asyncio.run(backend._init_store())

    assert isinstance(st, backend.InMemoryStore), (
        "建表失败也该拿到能用的内存 store，不能是 None"
    )
    assert got_pool is None
    assert pool.closed and pool.waited, "建表失败后池没被收掉，连接泄漏了"


def test_health_tells_apart_mysql_probe_and_store(monkeypatch):
    """mysql 探针说 ok、store 说 memory：库是活的，但 store 的表没建起来。
    两个字段各说各的，且退化不参与 overall 判定。"""
    monkeypatch.setattr(backend, "store", backend.InMemoryStore())
    body = _health(redis={"status": "ok", "active_sessions": 1},
                   mcp={"status": "connected"}).json()

    assert body["checks"]["mysql"]["status"] == "ok"
    assert body["checks"]["store"]["status"] == "memory"
    assert "detail" in body["checks"]["store"], "退化时要说明退到哪去了"
    assert body["status"] == "ok", "store 退化不该影响 overall"


def test_health_reports_mysql_backed_store(monkeypatch):
    """store 建起来时，状态要明确说是 MySQL 撑的，而不是笼统的 ok。"""
    monkeypatch.setattr(backend, "store", _real_mysql_store())
    body = _health(redis={"status": "ok", "active_sessions": 1},
                   mcp={"status": "connected"}).json()

    assert body["checks"]["store"] == {"status": "ok", "backend": "mysql"}


def test_health_reports_checkpointer_and_store_independently(monkeypatch):
    """两个组件分开报，不能用一个的状态去代表另一个。

    它们各自独立的池、各自独立降级，所以"一个成了另一个没成"是真实可能的
    （比如 store 的迁移脚本失败，而 checkpoint 的表早就建好了）。合成一个字段
    报的话，这种半边坏掉的情况就会被掩盖成"一切正常"。
    """
    monkeypatch.setattr(backend, "checkpointer", object())           # 对话记忆建起来了
    monkeypatch.setattr(backend, "store", backend.InMemoryStore())   # 长期记忆没起来
    body = _health(redis={"status": "ok", "active_sessions": 1},
                   mcp={"status": "connected"}).json()

    assert body["checks"]["checkpointer"] == {"status": "ok", "backend": "mysql"}
    assert body["checks"]["store"]["status"] == "memory"
