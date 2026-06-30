"""Redis 分布式锁（M4 核心）。

锁的正确性 = 三要素，缺一不可：
  1. **原子获取 + TTL**：SET key <token> NX PX <ttl> 一条命令搞定"不存在才设 + 设过期"。
     有 TTL 才能在持锁进程崩溃时自动释放（否则死锁）；原子才能避免"SET 后再 EXPIRE"
     两步之间崩溃留下无 TTL 的死锁。
  2. **唯一令牌(token)**：每次获取生成随机 token 写进 value，证明"这把锁是我的"。
  3. **比对后再删(Lua 原子)**：释放时只有 token 匹配才删——杜绝经典误删：
       A 拿锁(TTL 10s) → 活超时到 12s → 第 10s 锁过期、B 抢到 → A 第 12s 来 DEL，
       裸 DEL 会删掉【B 的锁】。Lua 里 `get==我的token 才 del`，A 拿旧 token 比不中 → 空操作。
     必须 Lua（服务端原子），而非应用层 get 完再 del——后者两步之间仍有过期窗口。

TTL 两难（太短失正确性 / 太长失可用性）用**看门狗(watchdog)**破：持锁期间后台周期性续期
（同样 Lua 校验"还是我的锁才续"），活干完停掉。auto_renew=True 开启。

注意边界（面试必答）：分布式锁是**建议性优化**，不是绝对正确性保证。持锁者若发生 GC
长暂停，醒来时锁可能已过期、别人已接管，它却继续写 → 脏。根治要**fencing token**
（单调递增、由被保护资源校验拒绝迟到者）。本实现是单实例 Redis 锁，主从切换时仍可能丢锁
（异步复制）；要更强一致需 Redlock + fencing，取舍见类文档。
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType
from uuid import uuid4

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("app.infra.locks")

# 释放：仅当锁还是我的（token 匹配）才删。原子，避免误删他人锁。
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# 续期：仅当锁还是我的才把过期时间重置为 ttl（看门狗用）。
_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""


class LockNotAcquired(Exception):
    """在超时时间内未能获取到锁。"""


def lock_key(tenant_id: str, name: str) -> str:
    """统一的锁 key 命名：带租户前缀（缓存层隔离），lock 命名空间。"""
    return f"t:{tenant_id}:lock:{name}"


class RedisLock:
    """基于 Redis 的分布式互斥锁，支持阻塞获取、Lua 安全释放、可选看门狗续期。

    既可显式 acquire/release，也可用 `async with`（推荐，自动释放、自动管理看门狗）：

        async with RedisLock(redis, lock_key(tid, f"thread:{thread_id}"), auto_renew=True):
            ...  # 临界区

    获取不到（阻塞超时）时 `async with` 抛 LockNotAcquired，由调用方决定排队/拒绝/降级。
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        key: str,
        *,
        ttl_ms: int | None = None,
        blocking: bool = True,
        blocking_timeout: float | None = None,
        retry_interval: float | None = None,
        auto_renew: bool = False,
    ) -> None:
        self._redis = redis
        self._key = key
        self._token = uuid4().hex  # 唯一令牌：证明锁归属
        self._ttl_ms = ttl_ms if ttl_ms is not None else settings.lock_default_ttl_ms
        self._blocking = blocking
        self._blocking_timeout = (
            blocking_timeout if blocking_timeout is not None else settings.lock_blocking_timeout
        )
        self._retry_interval = (
            retry_interval if retry_interval is not None else settings.lock_retry_interval
        )
        self._auto_renew = auto_renew
        # 把 Lua 注册成脚本对象：首次 EVAL，之后走 EVALSHA（省带宽）
        self._release_script = redis.register_script(_RELEASE_LUA)
        self._renew_script = redis.register_script(_RENEW_LUA)
        self._wd_task: asyncio.Task | None = None

    @property
    def token(self) -> str:
        return self._token

    async def acquire(self) -> bool:
        """尝试获取锁。blocking=True 时轮询直到拿到或超时；False 时立即返回。

        返回 True=拿到，False=没拿到（非阻塞失败 / 阻塞超时）。
        """
        deadline = time.monotonic() + self._blocking_timeout
        while True:
            # SET NX PX：原子的"不存在才设 + 设过期"。拿到返回 True，已被占返回 None。
            ok = await self._redis.set(self._key, self._token, nx=True, px=self._ttl_ms)
            if ok:
                if self._auto_renew:
                    self._start_watchdog()
                return True
            if not self._blocking or time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._retry_interval)

    async def release(self) -> bool:
        """释放锁（Lua 比对 token，非属主则空操作）。返回是否真的删除了。"""
        await self._stop_watchdog()
        deleted = await self._release_script(keys=[self._key], args=[self._token])
        return bool(deleted)

    async def renew(self) -> bool:
        """续期（Lua 比对 token）。返回是否续上（False=锁已不是我的）。"""
        ok = await self._renew_script(keys=[self._key], args=[self._token, self._ttl_ms])
        return bool(ok)

    # —— 看门狗：持锁期间后台周期续期，破解 TTL 两难 ——

    def _start_watchdog(self) -> None:
        self._wd_task = asyncio.create_task(self._watchdog())

    async def _stop_watchdog(self) -> None:
        if self._wd_task is not None:
            self._wd_task.cancel()
            try:
                await self._wd_task
            except asyncio.CancelledError:
                pass
            self._wd_task = None

    async def _watchdog(self) -> None:
        # 每过 1/3 TTL 续一次：留足重试余量，又不过于频繁。
        interval = max(self._ttl_ms // 3, 1) / 1000
        try:
            while True:
                await asyncio.sleep(interval)
                if not await self.renew():
                    # 锁已不是我的（被抢/被删）——续期无意义，记日志后停。
                    log.warning("lock.watchdog_lost", key=self._key)
                    return
        except asyncio.CancelledError:
            raise

    async def __aenter__(self) -> RedisLock:
        if not await self.acquire():
            raise LockNotAcquired(self._key)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.release()
