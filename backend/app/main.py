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

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db import session as db
from app.infra import redis_client as rds

# 进程启动即配置日志（在任何 log 调用之前）
setup_logging()
log = get_logger("app.main")

# 探活超时：依赖卡住时不让 /health 一直挂着
_PROBE_TIMEOUT = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建立连接池，关闭时优雅释放。

    三个客户端都是"懒连接"的——创建对象/引擎时并不真正连，首次借连接才连，
    所以即使依赖还没起来，应用也能正常启动，由 /health 如实报告其状态（降级不阻断）。
    池/引擎全局只建一次，存进 app.state 供所有请求复用。
    """
    log.info("app.startup", app=settings.app_name, env=settings.app_env)
    # Postgres：SQLAlchemy 异步引擎（内含连接池）+ 会话工厂
    app.state.db_engine = db.build_engine()
    app.state.db_sessionmaker = db.build_sessionmaker(app.state.db_engine)
    # Redis：显式连接池 + client
    app.state.redis_pool = rds.build_redis_pool()
    app.state.redis = rds.build_redis_client(app.state.redis_pool)
    # Qdrant
    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    yield
    log.info("app.shutdown")
    await app.state.redis.aclose()
    await app.state.redis_pool.aclose()
    await app.state.qdrant.close()
    await app.state.db_engine.dispose()  # 关闭池里所有连接


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


async def _check_postgres() -> dict[str, Any]:
    try:
        await asyncio.wait_for(db.ping(app.state.db_engine), timeout=_PROBE_TIMEOUT)
        # 把连接池实时状态一并返回，让你能"看见"池在工作
        return {"ok": True, "pool": db.pool_stats(app.state.db_engine)}
    except Exception as e:  # noqa: BLE001 —— 探活就是要兜住所有异常如实上报
        return {"ok": False, "error": repr(e)}


async def _check_redis() -> dict[str, Any]:
    try:
        await asyncio.wait_for(app.state.redis.ping(), timeout=_PROBE_TIMEOUT)
        return {"ok": True, "pool": rds.pool_stats(app.state.redis_pool)}
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
