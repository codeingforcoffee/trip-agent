"""动态配置层（M9）：在 pydantic-settings 的【静态】配置之上，叠一个可选的【配置中心】热更新层。

先立设计立场（面试要能讲清"为什么这么分层"）：
  - **静态配置**（连接串、密钥、池参数）仍归 pydantic-settings —— 单一事实来源，不该运行时变；
  - 只有真正需要【热更新】的子集（CORS 白名单、限流阈值、功能开关）走这一层；
  - **可插拔 + 兜底**：默认只读 env（pydantic）；开了 Apollo 才叠加，且 **Apollo 不可达自动回退 env**
    —— 绝不因为配置中心挂了就拖垮应用（延续 M0 的"降级不阻断"）；
  - **不耦合具体客户端库**：Python 的 Apollo 客户端都是社区维护、质量参差，所以我们按 Apollo 的
    HTTP 协议自己实现一个最小客户端，藏在 ConfigSource 接口后面 —— 将来换 Nacos/Consul 只改一个 Source。

为什么 CORS 热更新需要【自定义中间件】而不是直接把 allow_origins 从 Apollo 读进 add_middleware：
  Starlette 的 CORSMiddleware 在【启动时】就把 allow_origins 定死了。"改了 Apollo 应用秒级生效、
  不重启"才是有含金量的部分 —— 得让中间件【每请求】去读当前白名单。做法是子类化 CORSMiddleware，
  只覆盖 is_allowed_origin() 这个判定钩子，其余 CORS 头逻辑（预检、Vary、凭证）全部复用，最小且正确。
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import time
from typing import Protocol

from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger("app.core.dynamic_config")


# ———————————————————————————— 配置源接口 ————————————————————————————
class ConfigSource(Protocol):
    """一个配置源：能按 key 取值（取不到返回 None，交给下一层兜底），并可选地起停后台刷新。"""

    def get(self, key: str) -> str | None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class EnvConfigSource:
    """env / pydantic-settings 源 —— 永远可用的【兜底层】。

    直接按属性名从 settings 取；列表（如 cors_origins）拼成逗号串，与 Apollo 里的存法对齐
    （Apollo 命名空间里同样用 `cors_origins=a,b,c` 的扁平 key→string）。
    """

    def get(self, key: str) -> str | None:
        val = getattr(settings, key, None)
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return ",".join(str(x) for x in val)
        return str(val)

    async def start(self) -> None:  # env 无需后台任务
        return None

    async def stop(self) -> None:
        return None


class ApolloConfigSource:
    """携程 Apollo 配置中心的【最小 HTTP 客户端】：拉取命名空间配置 + 后台轮询热更新。

    只依赖 Apollo 的两个能力，避免引第三方库：
      - 拉配置：GET {meta}/configs/{appId}/{cluster}/{namespace} → {"configurations": {k: v}}；
      - 热更新：这里用【定时轮询】（简单、稳）。Apollo 另有 notifications/v2 长轮询可做到近实时推送，
        属生产优化，留作 TODO（长轮询 60s 挂起、变更即返回，减少无效请求）。

    访问密钥（namespace 开了访问控制才需要）时，按 Apollo 文档做 HMAC-SHA1 签名。
    任何网络/解析异常都不抛，只告警 —— 由 DynamicConfig 回退到 env。
    """

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    async def start(self) -> None:
        self._stop.clear()
        try:
            await self._pull_once()
            log.info(
                "apollo.connected",
                app_id=settings.apollo_app_id,
                namespace=settings.apollo_namespace,
                keys=len(self._cache),
            )
        except Exception as e:  # noqa: BLE001 —— 首拉失败不阻断启动，回退 env
            log.warning("apollo.initial_pull_failed", error=repr(e))
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # —— 内部 ——
    def _config_path(self) -> str:
        return f"/configs/{settings.apollo_app_id}/{settings.apollo_cluster}/{settings.apollo_namespace}"

    def _signed_headers(self, path_with_query: str) -> dict[str, str]:
        """开了 secret 时按 Apollo 方案签名：sign = base64(hmac_sha1(secret, timestamp + '\\n' + path))。"""
        if not settings.apollo_secret:
            return {}
        ts = str(int(time.time() * 1000))
        string_to_sign = f"{ts}\n{path_with_query}"
        digest = hmac.new(
            settings.apollo_secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        sign = base64.b64encode(digest).decode("utf-8")
        return {"Authorization": f"Apollo {settings.apollo_app_id}:{sign}", "Timestamp": ts}

    async def _pull_once(self) -> None:
        import httpx  # 懒导入：只有开了 Apollo 才需要，未启用则零成本

        path = self._config_path()
        url = f"{settings.apollo_meta.rstrip('/')}{path}"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=self._signed_headers(path))
            resp.raise_for_status()
            data = resp.json()
        new = {str(k): str(v) for k, v in (data.get("configurations") or {}).items()}
        if new != self._cache:
            log.info("apollo.config_changed", keys=sorted(new.keys()))
        self._cache = new

    async def _poll_loop(self) -> None:
        interval = settings.apollo_poll_interval
        while not self._stop.is_set():
            try:
                # 等一个轮询间隔，或提前被 stop 唤醒（可及时优雅退出）
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                break  # stop 被 set
            except asyncio.TimeoutError:
                pass
            try:
                await self._pull_once()
            except Exception as e:  # noqa: BLE001 —— 轮询失败保留上次缓存，不炸
                log.warning("apollo.poll_failed", error=repr(e))


# ———————————————————————————— 动态配置聚合 ————————————————————————————
class DynamicConfig:
    """按优先级叠加多个配置源：Apollo（若启用）在前、env 永远兜底。

    单例（模块底部 dynamic_config），像 settings 一样全局取用。启动前就能用（只有 env），
    lifespan 里 start() 后才接通 Apollo —— 中间件持有的是同一个单例，无需重新注册。
    """

    def __init__(self) -> None:
        self._env = EnvConfigSource()
        self._apollo: ApolloConfigSource | None = None
        self._sources: list[ConfigSource] = [self._env]  # 启动前只有 env 兜底

    async def start(self) -> None:
        if settings.apollo_enabled:
            self._apollo = ApolloConfigSource()
            await self._apollo.start()
            self._sources = [self._apollo, self._env]  # Apollo 优先，env 兜底
            log.info("dynamic_config.apollo_enabled")
        else:
            log.info("dynamic_config.env_only")

    async def stop(self) -> None:
        if self._apollo is not None:
            await self._apollo.stop()

    def get(self, key: str, default: str | None = None) -> str | None:
        for src in self._sources:
            val = src.get(key)
            if val is not None:
                return val
        return default

    def allow_origins(self) -> list[str]:
        """当前 CORS 白名单（逗号分隔 → 列表）。每次调用都读实时值，故支持热更新。"""
        raw = self.get("cors_origins", "") or ""
        return [o.strip() for o in raw.split(",") if o.strip()]


# 全局单例（与 settings 同风格）
dynamic_config = DynamicConfig()


# ———————————————————————————— 热更新 CORS 中间件 ————————————————————————————
class DynamicCORSMiddleware(CORSMiddleware):
    """CORS 中间件的热更新版：白名单每请求从 DynamicConfig 实时读，改了 Apollo 立即生效、无需重启。

    只覆盖 is_allowed_origin() 这一个判定钩子，其余（预检 OPTIONS、Access-Control-* 头、Vary: Origin、
    凭证处理）全部复用父类 —— 既拿到热更新，又不重造易错的 CORS 逻辑。
    构造时传 allow_origins=[]（走"显式源"模式，实际判定交给下面的覆盖方法）。
    """

    def __init__(self, app: ASGIApp, config: DynamicConfig, **kwargs) -> None:
        self._dyn = config
        super().__init__(app, allow_origins=[], **kwargs)

    def is_allowed_origin(self, origin: str) -> bool:
        return origin in self._dyn.allow_origins()
