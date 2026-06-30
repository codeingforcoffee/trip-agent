"""工具调用的可靠性封装（M4）：超时 + 重试(仅幂等) + 熔断。

叠加在 M2 的工具层之上，不改图结构。三件事，对应分布式可靠性三个经典手段：

  1. **超时（timeout）**：asyncio.wait_for 包住调用。下游卡死时快速失败，不让一个
     慢工具拖垮整轮 fan-out。
  2. **重试 + 退避抖动（retry/backoff/jitter）**：瞬时故障自动重试，指数退避避免雪崩，
     随机抖动避免"惊群同步重试"。**关键：只对幂等/只读工具重试**——非幂等(下单)盲目
     重试会重复扣款，所以非幂等只试一次（呼应"超时是模糊的，不知道到底成没成功"）。
  3. **熔断（circuit breaker）**：某工具连续失败超阈值就"开路"，一段冷却期内直接快速失败、
     不再打那个已经在挂的下游；冷却后"半开"放一个探测请求，成功则恢复。保护下游、也避免
     调用方把时间全耗在必然失败的等待上。

熔断状态是**进程内**的（每个实例保护自己）。要跨实例共享需放 Redis，这里从简。
参数校验错误(ValidationError)不算下游故障：不重试、不计入熔断，直接抛回让模型自纠(M2 行为)。
"""

from __future__ import annotations

import asyncio
import random
import time

from pydantic import ValidationError

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("app.agent.reliability")

# 非幂等工具名单：对这些工具【不自动重试】。M7 的下单/取消会 mark_non_idempotent 登记进来。
_NON_IDEMPOTENT: set[str] = set()


def mark_non_idempotent(name: str) -> None:
    _NON_IDEMPOTENT.add(name)


def is_idempotent(name: str) -> bool:
    return name not in _NON_IDEMPOTENT


class CircuitOpen(Exception):
    """熔断开路：工具暂时不可用，调用方应降级（而非等待必然失败）。"""


class CircuitBreaker:
    """三态熔断器：closed（正常）→ open（开路，快速失败）→ half_open（放一个探测）。"""

    def __init__(self, fail_threshold: int, cooldown_s: float) -> None:
        self._threshold = fail_threshold
        self._cooldown = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None
        self.state = "closed"

    def allow(self) -> bool:
        """是否放行本次调用。open 且冷却已过 → 转 half_open 放一个探测。"""
        if self.state == "open":
            if time.monotonic() - (self._opened_at or 0) >= self._cooldown:
                self.state = "half_open"
                return True
            return False
        return True  # closed / half_open

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self.state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        # 半开探测又失败、或失败累计到阈值 → 开路
        if self.state == "half_open" or self._failures >= self._threshold:
            self.state = "open"
            self._opened_at = time.monotonic()


_breakers: dict[str, CircuitBreaker] = {}


def _breaker(name: str) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(
            settings.breaker_fail_threshold, settings.breaker_cooldown_s
        )
    return _breakers[name]


def reset_breakers() -> None:
    """清空熔断状态（测试用，保证用例间互不干扰）。"""
    _breakers.clear()


def _backoff(attempt: int, base: float) -> float:
    """指数退避 + 满抖动(full jitter)：base*2^attempt 的基础上叠加随机量，打散重试时刻。"""
    return base * (2**attempt) + random.uniform(0, base)


async def call_tool_resilient(
    tool,
    args: dict,
    *,
    name: str,
    config: dict | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
) -> str:
    """带超时/重试/熔断地调用一个工具，返回结果字符串。

    config：LangGraph 的 RunnableConfig（含 configurable.tenant_id 等身份）。传入时透传给
    tool.ainvoke，工具（如 M5 的 RAG policy）据此做租户过滤；为 None 时按单参调用（兼容离线 stub）。

    抛出：CircuitOpen（开路，调用方降级）、ValidationError（参数错，模型自纠）、
    或最后一次的原始异常（重试耗尽）。
    """
    timeout = timeout if timeout is not None else settings.tool_timeout_s
    max_retries = max_retries if max_retries is not None else settings.tool_max_retries
    base_delay = base_delay if base_delay is not None else settings.tool_retry_base_delay

    breaker = _breaker(name)
    if not breaker.allow():
        raise CircuitOpen(name)

    attempts = (max_retries + 1) if is_idempotent(name) else 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            # 有 config 就透传（真工具按 config.configurable 取租户）；没有则单参调用（stub/离线）
            coro = tool.ainvoke(args, config) if config is not None else tool.ainvoke(args)
            result = await asyncio.wait_for(coro, timeout)
            breaker.record_success()
            return str(result)
        except ValidationError:
            # 参数校验失败：不是下游故障，不重试、不计熔断，直接抛回去让模型修正参数
            raise
        except Exception as e:  # noqa: BLE001 —— 超时/下游异常都进重试+熔断逻辑
            last_exc = e
            breaker.record_failure()
            log.info(
                "tool.attempt_failed",
                tool=name,
                attempt=attempt + 1,
                attempts=attempts,
                error=repr(e),
            )
            if not breaker.allow():  # 刚刚熔断 → 别再徒劳重试
                break
            if attempt < attempts - 1:
                await asyncio.sleep(_backoff(attempt, base_delay))

    assert last_exc is not None
    raise last_exc
