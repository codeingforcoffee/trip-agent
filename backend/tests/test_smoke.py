"""M0 冒烟测试：验证配置解析 + /health 契约。

用 starlette 的 TestClient（同步）而非裸 httpx，是因为 TestClient 会触发
FastAPI 的 lifespan（startup/shutdown），从而正确创建 redis/qdrant 客户端。
依赖容器起着时 status=ok；没起时 status=degraded —— 两种都算"契约正确"，
所以这个测试不强依赖外部服务，可离线跑（这正是后续可离线评测的雏形）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_config_urls_resolve():
    assert settings.database_dsn.startswith("postgresql://")
    assert settings.redis_url.startswith("redis://")
    assert settings.qdrant_url.startswith("http://")


def test_root():
    with TestClient(app) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["app"] == settings.app_name


def test_health_contract():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        # 三个依赖都必须被探活并报告状态
        assert set(body["deps"]) == {"postgres", "redis", "qdrant"}
        for dep in body["deps"].values():
            assert "ok" in dep
