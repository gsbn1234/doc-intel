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

不碰网络、不碰真 Redis：外部依赖全换成桩。
"""
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
    assert set(body["checks"]) == {"redis", "session_dir", "mcp", "mysql", "checkpointer"}
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

    with patch.object(backend, "close_mcp", fake_close):
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
