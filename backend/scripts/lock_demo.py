"""分布式锁并发演示（M4）：眼见为实。

运行：cd backend && uv run python scripts/lock_demo.py（需 make up 起 Redis）

演示四件事：
  1. 抢锁互斥：100 个并发非阻塞抢同一把锁 → 只有 1 个成功；
  2. 临界区保护：不加锁的并发自增会丢更新，加锁后结果正确；
  3. TTL 自动释放：持锁者"崩溃"（不释放）→ 别人等 TTL 过后能拿到；
  4. 看门狗续期：持锁超过 TTL，看门狗续着 → 期间竞争者拿不到。
"""

from __future__ import annotations

import asyncio

import redis.asyncio as aioredis

from app.core.config import settings
from app.infra.locks import RedisLock, lock_key

TID = "demo-tenant"


async def demo_mutual_exclusion(redis) -> None:
    key = lock_key(TID, "mutex")
    await redis.delete(key)

    async def contender() -> bool:
        lock = RedisLock(redis, key, ttl_ms=2000, blocking=False)
        got = await lock.acquire()
        if got:
            await asyncio.sleep(0.2)  # 持锁干点活
            await lock.release()
        return got

    results = await asyncio.gather(*(contender() for _ in range(100)))
    print(f"1) 互斥：100 个并发抢锁，成功 {sum(results)} 个（期望 1）")


async def demo_critical_section(redis) -> None:
    key = lock_key(TID, "counter")
    await redis.delete(key)

    class Counter:
        value = 0

    async def incr_unsafe(c: Counter) -> None:
        v = c.value  # 读
        await asyncio.sleep(0)  # 让出事件循环，制造交错
        c.value = v + 1  # 写（read-modify-write 非原子）

    async def incr_safe(c: Counter) -> None:
        async with RedisLock(redis, key, ttl_ms=5000, blocking_timeout=10):
            v = c.value
            await asyncio.sleep(0)
            c.value = v + 1

    unsafe = Counter()
    await asyncio.gather(*(incr_unsafe(unsafe) for _ in range(200)))
    safe = Counter()
    await asyncio.gather(*(incr_safe(safe) for _ in range(200)))
    print(
        f"2) 临界区：200 次并发自增 —— 不加锁={unsafe.value}（丢更新）, 加锁={safe.value}（期望 200）"
    )


async def demo_ttl_auto_release(redis) -> None:
    key = lock_key(TID, "ttl")
    await redis.delete(key)
    # 持锁者"崩溃"：拿到锁后不释放，只给 600ms TTL
    crashed = RedisLock(redis, key, ttl_ms=600, blocking=False)
    await crashed.acquire()
    # 立刻抢 → 抢不到
    now = RedisLock(redis, key, ttl_ms=600, blocking=False)
    got_now = await now.acquire()
    # 等过 TTL 再抢 → 抢得到（锁自动释放，没死锁）
    await asyncio.sleep(0.8)
    later = RedisLock(redis, key, ttl_ms=600, blocking=False)
    got_later = await later.acquire()
    await later.release()
    print(
        f"3) TTL 自动释放：崩溃后立刻抢={got_now}（期望 False）, 过 TTL 再抢={got_later}（期望 True）"
    )


async def demo_watchdog(redis) -> None:
    key = lock_key(TID, "watchdog")
    await redis.delete(key)
    # 短 TTL + 看门狗：持锁 1.5s，远超 600ms TTL，但看门狗续着不丢
    holder = RedisLock(redis, key, ttl_ms=600, blocking=False, auto_renew=True)
    await holder.acquire()
    await asyncio.sleep(1.5)  # 期间看门狗每 200ms 续一次
    competitor = RedisLock(redis, key, ttl_ms=600, blocking=False)
    got = await competitor.acquire()  # 应失败：锁被看门狗续着
    await holder.release()
    print(
        f"4) 看门狗续期：持锁 1.5s（TTL 仅 600ms），期间竞争者抢锁={got}（期望 False，看门狗保住了锁）"
    )


async def main() -> None:
    # 演示要 100+ 协程同时抢锁，瞬时连接需求高，单独建个大池（生产应用池是 50，
    # 高并发场景要么调大、要么用 BlockingConnectionPool 让请求排队等连接而非报错）。
    pool = aioredis.ConnectionPool.from_url(
        settings.redis_url, max_connections=256, decode_responses=True
    )
    redis = aioredis.Redis(connection_pool=pool)
    try:
        await redis.ping()
    except Exception as e:  # noqa: BLE001
        print(f"需要 Redis（make up）：{e!r}")
        await redis.aclose()
        await pool.aclose()
        return
    print(f"Redis: {settings.redis_url}\n")
    await demo_mutual_exclusion(redis)
    await demo_critical_section(redis)
    await demo_ttl_auto_release(redis)
    await demo_watchdog(redis)
    await redis.aclose()
    await pool.aclose()


if __name__ == "__main__":
    asyncio.run(main())
