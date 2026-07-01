"""刷新令牌的服务端家族记录（M9b）：让无状态的 refresh JWT 变得【可旋转 + 可撤销 + 可检测盗用】。

为什么只存"一条记录 / 会话"就够：
  一次登录 = 一个令牌家族 family。服务端在 Redis 只存 `t:{tenant}:rtfam:{family}` = 【当前有效 jti】。
  - 旋转（正常刷新）：jti 匹配 → 生成新 jti、写回，旧令牌（旧 jti）自然失效；
  - 撤销（登出）：删掉这条 key，整条会话作废；
  - **盗用检测**：若来的 refresh 令牌 jti 与记录不符（= 一个已被旋转掉的旧令牌被重放，典型是令牌被盗后
    攻击者与真实用户各刷一次）→ 判定盗用，**删掉整个家族**，双方后续都失效，逼迫重新登录。

原子性：get→比对→set/del 必须原子，否则并发刷新会竞态（两个请求同时读到同一 jti 各自旋转）。
用 Lua 脚本在 Redis 端一次跑完（与 M4 分布式锁的安全释放同一手法：register_script + EVALSHA）。

带租户前缀 `t:{tenant_id}:` —— 与 M4 锁/限流、多租户缓存命名空间约定一致。
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.core.config import settings

# 旋转 + 盗用检测，原子完成。
#   KEYS[1] = 家族 key；ARGV[1] = 期望的当前 jti；ARGV[2] = 新 jti；ARGV[3] = ttl 秒
#   返回：'ok'（已旋转）/ 'reuse'（检测到重放，已作废家族）/ 'revoked'（家族不存在=已登出/过期）
_ROTATE_LUA = """
local cur = redis.call('GET', KEYS[1])
if not cur then
    return 'revoked'
end
if cur ~= ARGV[1] then
    redis.call('DEL', KEYS[1])
    return 'reuse'
end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 'ok'
"""


def family_key(tenant_id: str, family: str) -> str:
    return f"t:{tenant_id}:rtfam:{family}"


class RefreshStore:
    """刷新令牌家族记录（Redis）。TTL 天然承载过期，删除即撤销。"""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._rotate = redis.register_script(_ROTATE_LUA)

    def _ttl_seconds(self) -> int:
        return settings.jwt_refresh_expire_days * 24 * 3600

    async def register(self, tenant_id: str, family: str, jti: str) -> None:
        """登录时登记新家族的当前 jti（TTL = 刷新令牌有效期）。"""
        await self._redis.set(family_key(tenant_id, family), jti, ex=self._ttl_seconds())

    async def rotate(self, tenant_id: str, family: str, expected_jti: str, new_jti: str) -> str:
        """原子旋转：返回 'ok' / 'reuse' / 'revoked'（见 _ROTATE_LUA）。"""
        result = await self._rotate(
            keys=[family_key(tenant_id, family)],
            args=[expected_jti, new_jti, self._ttl_seconds()],
        )
        # redis-py 开了 decode_responses，脚本返回值可能是 str 或 bytes，统一成 str
        return result.decode() if isinstance(result, bytes) else str(result)

    async def revoke(self, tenant_id: str, family: str) -> None:
        """登出/主动撤销：删掉家族记录，整条会话作废。"""
        await self._redis.delete(family_key(tenant_id, family))
