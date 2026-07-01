"""M9 动态配置层单测：hermetic —— 不连 Apollo/网络，用假源验证分层、兜底与【热更新语义】。

重点验证两件事：
  1. 优先级与兜底：Apollo 源命中则用之，取不到则回退 env，绝不因配置中心缺值而空；
  2. 热更新：allow_origins() / is_allowed_origin() 每次都读【实时】值，改了源立即反映（无需重启/重建）。
"""

from __future__ import annotations

from app.core.config import settings
from app.core.dynamic_config import (
    DynamicConfig,
    DynamicCORSMiddleware,
    EnvConfigSource,
)


class _FakeSource:
    """可变的假配置源：测试里直接改 store 即模拟"配置中心改了值"。"""

    def __init__(self, store: dict[str, str]) -> None:
        self.store = store

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def start(self) -> None:  # noqa: D401
        return None

    async def stop(self) -> None:
        return None


async def _dummy_app(scope, receive, send):  # 给中间件当下游用的空 ASGI app
    return None


# ————————————————————————— EnvConfigSource —————————————————————————


def test_env_source_joins_list_and_missing_key():
    env = EnvConfigSource()
    assert env.get("cors_origins") == ",".join(settings.cors_origins)
    assert env.get("nonexistent_key_xyz") is None


# ————————————————————————— DynamicConfig 分层/兜底 —————————————————————————


def test_dynamic_config_defaults_to_env():
    dc = DynamicConfig()  # 未 start：只有 env 兜底
    assert dc.allow_origins() == list(settings.cors_origins)


def test_dynamic_config_apollo_overrides_env():
    dc = DynamicConfig()
    apollo = _FakeSource({"cors_origins": "https://a.acme.com,https://b.acme.com"})
    dc._sources = [apollo, EnvConfigSource()]  # 模拟 start() 后的分层：Apollo 优先
    assert dc.allow_origins() == ["https://a.acme.com", "https://b.acme.com"]


def test_dynamic_config_falls_back_when_apollo_missing_key():
    dc = DynamicConfig()
    apollo = _FakeSource({})  # Apollo 里没有 cors_origins
    dc._sources = [apollo, EnvConfigSource()]
    # 该 key 取不到 → 回退 env，不空
    assert dc.allow_origins() == list(settings.cors_origins)


def test_dynamic_config_hot_reload():
    dc = DynamicConfig()
    apollo = _FakeSource({"cors_origins": "https://old.acme.com"})
    dc._sources = [apollo, EnvConfigSource()]
    assert dc.allow_origins() == ["https://old.acme.com"]
    # 模拟"在 Apollo Portal 改了白名单" → 下次读取立即生效，无需重启/重建对象
    apollo.store["cors_origins"] = "https://new.acme.com,https://extra.acme.com"
    assert dc.allow_origins() == ["https://new.acme.com", "https://extra.acme.com"]


# ————————————————————————— 热更新 CORS 中间件 —————————————————————————


def test_dynamic_cors_middleware_reads_live_whitelist():
    dc = DynamicConfig()
    apollo = _FakeSource({"cors_origins": "https://allowed.acme.com"})
    dc._sources = [apollo, EnvConfigSource()]
    mw = DynamicCORSMiddleware(_dummy_app, config=dc, allow_credentials=True)

    assert mw.is_allowed_origin("https://allowed.acme.com") is True
    assert mw.is_allowed_origin("https://evil.example.com") is False

    # 热更新：把新域名加进"配置中心" → 中间件下次判定立即放行（无需重建中间件/重启进程）
    apollo.store["cors_origins"] = "https://allowed.acme.com,https://evil.example.com"
    assert mw.is_allowed_origin("https://evil.example.com") is True
