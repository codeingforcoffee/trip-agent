"""Redis 客户端：配置驱动的连接池。

关键认知：redis-py 的 Redis / from_url 背后本身就是一个 ConnectionPool——
你以为的"一个 client"其实是"一个池"。这里我们把池参数显式配置出来
（而不是吃默认值），并做几件生产必须的事：
  - max_connections：池上限，防连接耗尽；
  - socket_timeout / socket_connect_timeout：网络抖动时快速失败，不无限挂；
  - health_check_interval：空闲连接定期自检，自动剔除被中间件掐断的死连接；
  - decode_responses：返回 str 而非 bytes，省去到处 .decode()。

M4 的分布式锁 / 限流都从这个池借连接。
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.core.config import settings


def build_redis_pool() -> aioredis.ConnectionPool:
    """按配置创建 Redis 连接池。整个进程只调一次。"""
    return aioredis.ConnectionPool.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        socket_timeout=settings.redis_socket_timeout,
        socket_connect_timeout=settings.redis_socket_connect_timeout,
        health_check_interval=settings.redis_health_check_interval,
        decode_responses=True,
    )


def build_redis_client(pool: aioredis.ConnectionPool) -> aioredis.Redis:
    """共用同一个池创建 client。可创建多个 client 共享一个池。"""
    return aioredis.Redis(connection_pool=pool)


def pool_stats(pool: aioredis.ConnectionPool) -> dict[str, int]:
    """读出池的实时状态，供 /health 展示。"""
    return {
        "max_connections": pool.max_connections,
        "in_use": len(pool._in_use_connections),  # 当前借出的连接数
        "available": len(pool._available_connections),  # 池中空闲可复用的连接数
    }
