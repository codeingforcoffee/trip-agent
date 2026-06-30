"""M4 集成测试：令牌桶限流（需要真实 Redis，未起则跳过）。"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.infra.ratelimit import RateLimiter, ratelimit_key
from app.infra.redis_client import build_redis_client, build_redis_pool


@pytest.fixture
async def redis():
    pool = build_redis_pool()
    client = build_redis_client(pool)
    try:
        await client.ping()
    except Exception as e:  # noqa: BLE001
        await client.aclose()
        await pool.aclose()
        pytest.skip(f"需要 Redis（make up）：{e!r}")
    yield client
    await client.aclose()
    await pool.aclose()


def _key() -> str:
    return ratelimit_key("test", "user", uuid4().hex)


async def test_capacity_is_hard_limit(redis):
    """补充极慢时，桶容量就是硬上限：容量 5 → 前 5 个放行，其余拒绝。"""
    rl = RateLimiter(redis, capacity=5, refill_per_sec=0.001)
    key = _key()
    results = [await rl.check(key) for _ in range(8)]
    assert sum(r.allowed for r in results) == 5
    assert results[-1].allowed is False
    assert results[-1].retry_after > 0  # 被拒时给出建议等待


async def test_refill_over_time(redis):
    """耗尽后等一会儿，令牌按速率补回，又能放行。"""
    rl = RateLimiter(redis, capacity=2, refill_per_sec=50.0)
    key = _key()
    assert (await rl.check(key)).allowed is True
    assert (await rl.check(key)).allowed is True
    assert (await rl.check(key)).allowed is False  # 桶空
    await asyncio.sleep(0.1)  # 50/s × 0.1s ≈ 5 个令牌补回（封顶 2）
    assert (await rl.check(key)).allowed is True  # 补回了


async def test_buckets_are_independent_per_key(redis):
    """不同 key（不同用户/租户）各自一个桶，互不影响。"""
    rl = RateLimiter(redis, capacity=1, refill_per_sec=0.001)
    k1, k2 = _key(), _key()
    assert (await rl.check(k1)).allowed is True
    assert (await rl.check(k1)).allowed is False  # k1 用光
    assert (await rl.check(k2)).allowed is True  # k2 不受影响
