"""可观测中间件（M9d）：给每个请求打 trace_id、计时、并核算 token 成本。

为什么用【纯 ASGI 中间件】而不是 Starlette 的 BaseHTTPMiddleware？关键在【流式响应】：
  /chat 是 SSE 长响应，token 总数要等整条流吐完才知道。BaseHTTPMiddleware 的
  `response = await call_next()` 对流式响应会【提前返回】（body 尚未发送），此刻既拿不到
  最终 token 数、也没法把时延测到"流真正结束"。而纯 ASGI 的 `await self.app(scope,…)`
  只在【整个响应发完】后才返回——于是我们能在其后读到累计用量、并算出这一整条流的真实时延。

用量如何跨"中间件 ↔ 流生成器"传递：用一个 ContextVar 存【可变账本(dict)】。中间件在请求入口
  set 一本空账；业务侧（/chat 流结束时）用 record_usage() 往【同一个 dict】写数（改内容、
  不重绑变量）。纯 ASGI 下 self.app(...) 与其内部的流式 body 跑在同一 asyncio 任务、共享同一
  Context，所以 app 返回后中间件读到的就是被写入后的值。

trace_id 的信任边界：可采信客户端传入的 X-Request-ID 以便端到端串联，但必须校验它是合法 UUID
  ——否则攻击者能借它往 JSON 日志里塞伪造字段/换行（log injection）。非法就自己生成一个。
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar

import structlog

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("app.http")

# 每请求一本"用量账本"。默认 None：非请求上下文（如后台任务）读到 None 即知"当前无账本"，
# record_usage 便安静跳过，不会误记到别处。
_usage_var: ContextVar[dict | None] = ContextVar("usage_ledger", default=None)


def record_usage(
    *, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int | None = None
) -> None:
    """业务侧把本请求的 token 用量记进当前账本（如 /chat 流结束时调用）。

    改字典内容而非重绑 ContextVar —— 这样中间件持有的同一个 dict 引用能看到更新。
    total_tokens 缺省时按 输入+输出 求和（个别模型不回总数）。
    """
    ledger = _usage_var.get()
    if ledger is None:  # 不在被中间件包裹的请求里（例如单测直接调），静默跳过
        return
    ledger["input_tokens"] += int(input_tokens or 0)
    ledger["output_tokens"] += int(output_tokens or 0)
    ledger["total_tokens"] += (
        int(total_tokens) if total_tokens is not None else int(input_tokens or 0) + int(output_tokens or 0)
    )


def estimate_cost_cny(input_tokens: int, output_tokens: int) -> float:
    """按配置价目表估算成本（人民币元）。输入/输出分开计价（差价常达数倍）。"""
    cost = (
        input_tokens / 1_000_000 * settings.deepseek_price_in_per_1m
        + output_tokens / 1_000_000 * settings.deepseek_price_out_per_1m
    )
    return round(cost, 6)


def _resolve_trace_id(headers: dict[str, str]) -> str:
    """取客户端 X-Request-ID（须为合法 UUID，防日志注入），否则新生成一个。"""
    raw = headers.get("x-request-id")
    if raw:
        try:
            return str(uuid.UUID(raw))  # 合法才采信；顺带归一化格式
        except ValueError:
            pass  # 非法一律丢弃，走下面自生成
    return uuid.uuid4().hex


class ObservabilityMiddleware:
    """纯 ASGI 中间件：trace_id 绑定 + 请求计时 + token/成本核算 + 结构化访问日志。

    挂在最外层（main.py 最后 add）：先于一切绑定 trace_id、并把整条链路（含内层中间件与
    流式 body）都计入时延。tenant_id/user_id 在鉴权成功后由 deps 补绑——那时才知道"是谁"。
    """

    def __init__(self, app):  # noqa: ANN001
        self.app = app

    async def __call__(self, scope, receive, send):  # noqa: ANN001
        if scope["type"] != "http":  # websocket/lifespan 直接放行
            await self.app(scope, receive, send)
            return

        # scope["headers"] 是 list[tuple[bytes, bytes]]
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        trace_id = _resolve_trace_id(headers)

        # 每请求一份干净的日志上下文：先清（防连接复用时跨请求串味），再绑 trace_id。
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)

        ledger = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        tok = _usage_var.set(ledger)

        status_holder = {"code": 500}

        async def send_wrapper(message):  # noqa: ANN001
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                # 回写 X-Request-ID：前端 / 网关 / 日志据此串联全链路
                message.setdefault("headers", []).append((b"x-request-id", trace_id.encode()))
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)  # 流式响应发完才返回
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            cost = estimate_cost_cny(ledger["input_tokens"], ledger["output_tokens"])
            log.info(
                "http.request",
                method=scope.get("method"),
                path=scope.get("path"),
                status=status_holder["code"],
                latency_ms=elapsed_ms,
                input_tokens=ledger["input_tokens"],
                output_tokens=ledger["output_tokens"],
                total_tokens=ledger["total_tokens"],
                cost_cny=cost,
            )
            _usage_var.reset(tok)
            structlog.contextvars.clear_contextvars()
