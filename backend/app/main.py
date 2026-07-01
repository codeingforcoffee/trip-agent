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
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from qdrant_client import AsyncQdrantClient

from app.agent.checkpointer import build_checkpointer_pool
from app.api import auth as auth_api
from app.api import chat as chat_api
from app.api import conversations as conversations_api
from app.core.config import settings
from app.core.dynamic_config import DynamicCORSMiddleware, dynamic_config
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
    # M9：短期记忆 checkpointer（连接池版）+ 懒构建的 Agent 图。
    #   - 图本身懒构建（首个 /chat 请求，见 api/chat._get_graph）：图依赖 LLM，缺 key
    #     不该阻断应用启动（延续 M0“降级不阻断”）；
    #   - checkpointer 依赖 Postgres（核心依赖），此处建池并建表。open(wait=False) 后台填连接、
    #     不阻塞启动；setup() 建表失败（如 Postgres 未起）只告警，不炸启动。
    app.state.agent_graph = None
    app.state.checkpointer = None
    app.state.checkpointer_pool = build_checkpointer_pool()
    await app.state.checkpointer_pool.open(wait=False)
    try:
        saver = AsyncPostgresSaver(app.state.checkpointer_pool)
        await saver.setup()
        app.state.checkpointer = saver
        log.info("app.checkpointer_ready")
    except Exception as e:  # noqa: BLE001 —— 依赖未就绪不阻断启动，由 /health 与首个 /chat 暴露
        log.warning("app.checkpointer_setup_failed", error=repr(e))
    # 动态配置层（M9）：启用 Apollo 时在此接通并起后台热更新轮询；否则只用 env（零成本）。
    await dynamic_config.start()
    yield
    log.info("app.shutdown")
    await dynamic_config.stop()
    await app.state.redis.aclose()
    await app.state.redis_pool.aclose()
    await app.state.qdrant.close()
    await app.state.checkpointer_pool.close()
    await app.state.db_engine.dispose()  # 关闭池里所有连接


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

# CORS（M9）：用【热更新版】中间件——白名单每请求从 dynamic_config 实时读（默认 env 兜底，
# 开了 Apollo 则由配置中心覆盖且改动秒级生效、无需重启）。allow_credentials=True 让前端能带凭证。
app.add_middleware(
    DynamicCORSMiddleware,
    config=dynamic_config,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 业务路由（M3）：登录发 JWT + 租户态会话增查
app.include_router(auth_api.router)
app.include_router(conversations_api.router)
# 流式对话（M9a）：SSE /chat + /chat/resume
app.include_router(chat_api.router)


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
