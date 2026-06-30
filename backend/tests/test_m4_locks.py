"""M4 集成测试：Redis 分布式锁（需要真实 Redis，未起则整组跳过）。

锁的语义无法离线测（Lua 原子性、TTL、看门狗都依赖真 Redis），所以这里连真库，
连不上自动 skip——与 M3 的集成测试一致。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.infra.locks import LockNotAcquired, RedisLock, lock_key
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
    return lock_key("test", f"m4-{uuid4().hex}")


async def test_only_one_of_many_acquires(redis):
    """20 个并发非阻塞抢同一把锁 → 恰好 1 个成功。"""
    key = _key()

    async def contend() -> bool:
        lock = RedisLock(redis, key, ttl_ms=3000, blocking=False)
        if await lock.acquire():
            await asyncio.sleep(0.1)
            await lock.release()
            return True
        return False

    results = await asyncio.gather(*(contend() for _ in range(20)))
    assert sum(results) == 1


async def test_release_only_by_owner(redis):
    """非属主释放是空操作（防误删）：A 持锁，B 拿不同 token 去释放删不掉。"""
    key = _key()
    a = RedisLock(redis, key, ttl_ms=5000, blocking=False)
    b = RedisLock(redis, key, ttl_ms=5000, blocking=False)
    assert await a.acquire() is True
    assert await b.acquire() is False  # 没抢到
    # B 用自己的 token 去释放 → 删不掉 A 的锁
    assert await b.release() is False
    # A 的锁还在 → 第三者仍抢不到
    c = RedisLock(redis, key, ttl_ms=5000, blocking=False)
    assert await c.acquire() is False
    # A 自己释放 → 成功
    assert await a.release() is True
    assert await c.acquire() is True
    await c.release()


async def test_ttl_auto_release_no_deadlock(redis):
    """持锁者"崩溃"（不释放）→ TTL 过后锁自动释放，不死锁。"""
    key = _key()
    crashed = RedisLock(redis, key, ttl_ms=400, blocking=False)
    assert await crashed.acquire() is True
    # 立刻抢拿不到
    assert await RedisLock(redis, key, blocking=False).acquire() is False
    # 等过 TTL
    await asyncio.sleep(0.6)
    later = RedisLock(redis, key, ttl_ms=400, blocking=False)
    assert await later.acquire() is True
    await later.release()


async def test_context_manager_acquires_and_releases(redis):
    key = _key()
    async with RedisLock(redis, key, ttl_ms=3000):
        assert await redis.get(key) is not None  # 锁在
    assert await redis.get(key) is None  # 退出后自动释放


async def test_blocking_timeout_raises(redis):
    """拿不到锁、阻塞超时 → __aenter__ 抛 LockNotAcquired。"""
    key = _key()
    holder = RedisLock(redis, key, ttl_ms=5000, blocking=False)
    assert await holder.acquire() is True
    with pytest.raises(LockNotAcquired):
        async with RedisLock(redis, key, blocking=True, blocking_timeout=0.3):
            pass
    await holder.release()


async def test_watchdog_keeps_lock_alive(redis):
    """短 TTL + 看门狗：持锁超过 TTL 期间锁不丢，竞争者抢不到。"""
    key = _key()
    holder = RedisLock(redis, key, ttl_ms=400, blocking=False, auto_renew=True)
    assert await holder.acquire() is True
    await asyncio.sleep(1.0)  # 远超 400ms TTL，看门狗续着
    competitor = RedisLock(redis, key, ttl_ms=400, blocking=False)
    assert await competitor.acquire() is False  # 锁还在 → 抢不到
    await holder.release()
    # 释放后（看门狗停）竞争者能拿到
    assert await competitor.acquire() is True
    await competitor.release()
