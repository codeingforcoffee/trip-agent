"""M9d 生产化收尾单测：可观测中间件 + 安全响应头。全部 hermetic——直接以裸 ASGI 驱动中间件，
不起 HTTP server / 不连依赖。

重点验证那个关键设计：纯 ASGI 中间件 + ContextVar 账本，能在【流式响应发完后】读到 token 用量
（这正是 BaseHTTPMiddleware 做不到的地方）。
"""

from __future__ import annotations

import uuid

import structlog

from app.core.observability import (
    ObservabilityMiddleware,
    _resolve_trace_id,
    _usage_var,
    estimate_cost_cny,
    record_usage,
)
from app.core.security_headers import SecurityHeadersMiddleware

# ————————————————————————— 测试替身 / 驱动 —————————————————————————


class _StreamingApp:
    """内层 ASGI app：先发 response.start，再【分块】发 body，body 期间业务侧记账。

    这一步是重点——record_usage 发生在 http.response.start 之后、流还没发完时；能被外层中间件
    读到，正是"纯 ASGI 的 self.app(...) 等整条流发完才返回 + 共享同一 Context"的证据。
    """

    def __init__(self, status: int = 200) -> None:
        self.status = status

    async def __call__(self, scope, receive, send):  # noqa: ANN001
        await send({"type": "http.response.start", "status": self.status, "headers": []})
        record_usage(input_tokens=100, output_tokens=50, total_tokens=150)
        await send({"type": "http.response.body", "body": b"chunk", "more_body": False})


def _http_scope(headers=None, scheme="http", method="POST", path="/chat"):  # noqa: ANN001
    hdrs = [(k.encode(), v.encode()) for k, v in (headers or {}).items()]
    return {"type": "http", "method": method, "path": path, "scheme": scheme, "headers": hdrs}


async def _drive(app, scope) -> list[dict]:  # noqa: ANN001
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):  # noqa: ANN001
        sent.append(message)

    await app(scope, receive, send)
    return sent


# ————————————————————————— 成本核算 / trace_id —————————————————————————


def test_estimate_cost_split_pricing():
    # 输入/输出分开计价：各 100 万 token 分别等于各自的单价（元/百万）。
    from app.core.config import settings

    assert estimate_cost_cny(1_000_000, 0) == round(settings.deepseek_price_in_per_1m, 6)
    assert estimate_cost_cny(0, 1_000_000) == round(settings.deepseek_price_out_per_1m, 6)


def test_resolve_trace_id_valid_passthrough():
    u = str(uuid.uuid4())
    assert _resolve_trace_id({"x-request-id": u}) == u  # 合法 UUID 采信


def test_resolve_trace_id_rejects_injection():
    # 信任边界：非法/含换行的 X-Request-ID 一律不采信，自生成——防日志注入。
    got = _resolve_trace_id({"x-request-id": "evil\ninjected field"})
    assert got != "evil\ninjected field"
    assert "\n" not in got


def test_resolve_trace_id_missing_generated():
    got = _resolve_trace_id({})
    assert got and "\n" not in got


def test_record_usage_without_ledger_is_silent():
    # 不在被中间件包裹的请求里（账本为 None）→ 静默跳过、不抛。
    assert _usage_var.get() is None
    record_usage(input_tokens=5, output_tokens=5)  # 不应抛异常


# ————————————————————————— ObservabilityMiddleware —————————————————————————


async def test_observability_reads_streamed_usage_and_sets_request_id(monkeypatch):
    """★核心★：流式 body 期间记的用量，中间件在流结束后能读到并算成本、写日志；且回写 X-Request-ID。"""
    captured: dict = {}

    class _FakeLog:
        def info(self, event, **kw):  # noqa: ANN001
            captured["event"] = event
            captured.update(kw)

    monkeypatch.setattr("app.core.observability.log", _FakeLog())

    app = ObservabilityMiddleware(_StreamingApp(status=200))
    sent = await _drive(app, _http_scope(path="/chat"))

    # 1) X-Request-ID 回写响应头
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert b"x-request-id" in dict(start["headers"])

    # 2) 中间件在流发完后读到账本（证明流式也拿得到 token 总数）并核算成本
    assert captured["event"] == "http.request"
    assert captured["input_tokens"] == 100
    assert captured["output_tokens"] == 50
    assert captured["total_tokens"] == 150
    assert captured["status"] == 200
    assert captured["path"] == "/chat"
    assert captured["cost_cny"] == estimate_cost_cny(100, 50)
    assert "latency_ms" in captured

    # 3) 请求结束后账本已重置、日志上下文已清（不跨请求串味）
    assert _usage_var.get() is None
    assert structlog.contextvars.get_contextvars() == {}


async def test_observability_uses_client_request_id(monkeypatch):
    """合法 X-Request-ID 应被采信并原样回写（端到端串联）。"""
    monkeypatch.setattr("app.core.observability.log", type("L", (), {"info": lambda *a, **k: None})())
    rid = str(uuid.uuid4())
    app = ObservabilityMiddleware(_StreamingApp())
    sent = await _drive(app, _http_scope(headers={"x-request-id": rid}))
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert dict(start["headers"])[b"x-request-id"] == rid.encode()


# ————————————————————————— SecurityHeadersMiddleware —————————————————————————


async def test_security_headers_present_and_no_hsts_on_http():
    app = SecurityHeadersMiddleware(_StreamingApp())
    sent = await _drive(app, _http_scope(scheme="http"))
    hdr = dict(next(m for m in sent if m["type"] == "http.response.start")["headers"])
    assert hdr[b"x-content-type-options"] == b"nosniff"
    assert hdr[b"x-frame-options"] == b"DENY"
    assert b"referrer-policy" in hdr
    assert b"strict-transport-security" not in hdr  # 明文 http 不发 HSTS


async def test_security_headers_hsts_only_on_https():
    app = SecurityHeadersMiddleware(_StreamingApp())
    sent = await _drive(app, _http_scope(scheme="https"))
    hdr = dict(next(m for m in sent if m["type"] == "http.response.start")["headers"])
    hsts = hdr[b"strict-transport-security"]
    assert b"max-age=" in hsts and b"includeSubDomains" in hsts
