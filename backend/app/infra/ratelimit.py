"""Redis 限流：令牌桶（M4）。

为什么令牌桶而非固定窗口：固定窗口在边界处会放进双倍突发（前窗末尾 + 后窗开头
挤在一起）。令牌桶按固定速率补充令牌、每请求取一个，**既允许可控突发**（桶容量）
**又长期受限**（补充速率），最适合 API 限流。

为什么用 Lua：判断+扣减必须**原子**完成。如果"读令牌数 → 判断 → 扣减"分成多条命令，
高并发下多个请求会读到同样的余量、各自都以为够 → 超发。Lua 在 Redis 服务端原子执行。

时间从应用侧传入（now_ms）而非用 Redis 的 TIME：可测试、确定性好；代价是多实例间依赖
各自时钟（需 NTP 同步）。要绝对一致可改用服务端 TIME。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import redis.asyncio as aioredis

# 令牌桶：先按流逝时间补充令牌（封顶 capacity），够则扣减放行，否则算出还要等多久。
# 状态存一个 hash：tokens（当前令牌数）、ts（上次补充时刻 ms）。返回 {放行?, 需等待 ms, 余量}。
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl_ms = tonumber(ARGV[5])

local d = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(d[1])
local ts = tonumber(d[2])
if tokens == nil then
    tokens = capacity
    ts = now_ms
end

local elapsed = now_ms - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(capacity, tokens + elapsed * refill_per_sec / 1000.0)

local allowed = 0
local retry_after_ms = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
else
    retry_after_ms = math.ceil((cost - tokens) / refill_per_sec * 1000)
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now_ms)
redis.call('PEXPIRE', key, ttl_ms)
return {allowed, retry_after_ms, math.floor(tokens)}
"""


def ratelimit_key(tenant_id: str, scope: str, ident: str) -> str:
    """限流 key：带租户前缀，rl 命名空间。scope 区分维度（user / tenant…）。"""
    return f"t:{tenant_id}:rl:{scope}:{ident}"


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after: float  # 秒，被拒时建议多久后再来（放进 Retry-After 头）
    remaining: int  # 桶里剩余令牌（近似，向下取整）


class RateLimiter:
    """令牌桶限流器。一个实例可服务多个 key（不同租户/用户各自一个桶）。"""

    def __init__(
        self,
        redis: aioredis.Redis,
        capacity: int,
        refill_per_sec: float,
    ) -> None:
        self._redis = redis
        self._capacity = capacity
        self._refill = refill_per_sec
        self._script = redis.register_script(_TOKEN_BUCKET_LUA)
        # 空闲桶的存活时间：够它从空补满的 2 倍，过后自动过期回收内存。
        self._ttl_ms = int(capacity / refill_per_sec * 1000 * 2) if refill_per_sec > 0 else 60_000

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        now_ms = int(time.time() * 1000)
        allowed, retry_ms, remaining = await self._script(
            keys=[key],
            args=[self._capacity, self._refill, now_ms, cost, self._ttl_ms],
        )
        return RateLimitResult(
            allowed=bool(allowed),
            retry_after=retry_ms / 1000.0,
            remaining=int(remaining),
        )
