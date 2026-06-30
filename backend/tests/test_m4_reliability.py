"""M4 离线单测：工具可靠性（超时 / 重试 / 熔断）。不连网，用 stub 工具驱动。"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app.agent.reliability import (
    CircuitBreaker,
    CircuitOpen,
    _breaker,
    call_tool_resilient,
    is_idempotent,
    mark_non_idempotent,
    reset_breakers,
)
from app.core.config import settings


@pytest.fixture(autouse=True)
def _clean_breakers():
    reset_breakers()
    yield
    reset_breakers()


class _StubTool:
    """假工具：按预设脚本决定每次调用成功返回还是抛异常。"""

    def __init__(self, *, fail_times: int = 0, sleep: float = 0.0, exc: Exception | None = None):
        self._fail_times = fail_times
        self._sleep = sleep
        self._exc = exc or RuntimeError("下游故障")
        self.calls = 0

    async def ainvoke(self, args):  # noqa: ANN001
        self.calls += 1
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self.calls <= self._fail_times:
            raise self._exc
        return "OK"


# ---------- 熔断器状态机（纯逻辑） ----------


def test_circuit_breaker_state_machine():
    b = CircuitBreaker(fail_threshold=2, cooldown_s=0.05)
    assert b.allow() and b.state == "closed"
    b.record_failure()
    assert b.allow()  # 1 次失败还没到阈值
    b.record_failure()  # 第 2 次 → 开路
    assert b.state == "open"
    assert b.allow() is False  # 开路期间快速失败
    import time

    time.sleep(0.06)  # 冷却过后
    assert b.allow() is True and b.state == "half_open"  # 放一个探测
    b.record_success()  # 探测成功 → 恢复
    assert b.state == "closed"


# ---------- call_tool_resilient ----------


async def test_idempotent_tool_retries_then_succeeds():
    tool = _StubTool(fail_times=2)  # 前 2 次失败，第 3 次成功
    out = await call_tool_resilient(tool, {}, name="search_x", max_retries=2, base_delay=0.001)
    assert out == "OK"
    assert tool.calls == 3  # 幂等 → 重试到成功


async def test_non_idempotent_not_retried():
    mark_non_idempotent("create_booking")
    assert is_idempotent("create_booking") is False
    tool = _StubTool(fail_times=5)
    with pytest.raises(RuntimeError):
        await call_tool_resilient(tool, {}, name="create_booking", max_retries=3, base_delay=0.001)
    assert tool.calls == 1  # 非幂等 → 只试一次，绝不重试


async def test_timeout_raises():
    tool = _StubTool(sleep=0.2)  # 比超时久
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await call_tool_resilient(tool, {}, name="slow_x", timeout=0.05, max_retries=0)


async def test_open_circuit_fails_fast_without_calling():
    # 先把某工具的熔断器手动打到开路
    b = _breaker("flaky_x")
    for _ in range(settings.breaker_fail_threshold):
        b.record_failure()
    assert b.state == "open"
    tool = _StubTool()
    with pytest.raises(CircuitOpen):
        await call_tool_resilient(tool, {}, name="flaky_x")
    assert tool.calls == 0  # 开路 → 根本不调用下游


async def test_validation_error_not_retried_not_tripped():
    class M(BaseModel):
        x: int

    class _BadArgs:
        calls = 0

        async def ainvoke(self, args):  # noqa: ANN001
            type(self).calls += 1
            M(x="not-an-int")  # 触发 pydantic ValidationError

    tool = _BadArgs()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await call_tool_resilient(tool, {}, name="val_x", max_retries=3, base_delay=0.001)
    assert tool.calls == 1  # 参数错不重试
    assert _breaker("val_x").state == "closed"  # 也不计入熔断
