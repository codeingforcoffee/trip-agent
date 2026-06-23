"""FastAPI 入口 + 健康探针。

M0 的核心交付物是 /health：它同时探活 Postgres / Redis / Qdrant 三个依赖，
是判断"环境是否就绪"的唯一事实来源。设计要点：
  - 探活并发执行（asyncio.gather），不串行等待；
  - 每个依赖各自 try/except，一个挂了不影响报告其它两个的状态；
  - 任一依赖不可用时返回 status=degraded（HTTP 仍 200，方便编排系统读 body 判断）。

后续里程碑会在这个 app 上挂载 /chat（M1/M9）、/auth（M3）等路由。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

# 进程启动即配置日志（在任何 log 调用之前）
setup_logging()
log = get_logger("app.main")

# 探活超时：依赖卡住时不让 /health 一直挂着
_PROBE_TIMEOUT = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建客户端，关闭时优雅释放。

    redis / qdrant 客户端是"懒连接"的——创建对象时并不真正连，首次用到才连，
    所以即使依赖还没起来，应用也能正常启动，由 /health 如实报告其状态。
    Postgres 我们在 M0 用"每次探活临时连一下"的方式（见 _check_postgres），
    到 M3 再换成 SQLAlchemy 连接池。
    """
    log.info("app.startup", app=settings.app_name, env=settings.app_env)
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    yield
    log.info("app.shutdown")
    await app.state.redis.aclose()
    await app.state.qdrant.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


async def _check_postgres() -> dict[str, Any]:
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(dsn=settings.database_dsn), timeout=_PROBE_TIMEOUT
        )
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()
        return {"ok": True}
    except Exception as e:  # noqa: BLE001 —— 探活就是要兜住所有异常如实上报
        return {"ok": False, "error": repr(e)}


async def _check_redis() -> dict[str, Any]:
    try:
        await asyncio.wait_for(app.state.redis.ping(), timeout=_PROBE_TIMEOUT)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


async def _check_qdrant() -> dict[str, Any]:
    try:
        await asyncio.wait_for(app.state.qdrant.get_collections(), timeout=_PROBE_TIMEOUT)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


@app.get("/")
async def root() -> dict[str, str]:
    return {"app": settings.app_name, "env": settings.app_env, "status": "running"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """并发探活三个依赖，返回每个的状态。"""
    pg, rd, qd = await asyncio.gather(_check_postgres(), _check_redis(), _check_qdrant())
    deps = {"postgres": pg, "redis": rd, "qdrant": qd}
    healthy = all(d["ok"] for d in deps.values())
    if not healthy:
        log.warning("health.degraded", deps=deps)
    return {"status": "ok" if healthy else "degraded", "deps": deps}
