"""Postgres 数据访问层：配置驱动的 SQLAlchemy 异步连接池。

为什么必须用连接池（而不是每次新建连接）？
  - 建连昂贵：TCP 握手 + TLS + Postgres 认证 + 后端进程 fork，毫秒级开销，
    高并发下会把延迟和 DB 进程数打爆；
  - 复用 + 限流：池复用连接，并给出"最多同时用多少连接"的硬上限，
    保护 DB（Postgres 的 max_connections 是稀缺资源，打满即雪崩）；
  - 自愈：pool_pre_ping 借出前探活，pool_recycle 定期回收，自动剔除死连接。

这个 engine 在应用生命周期里只创建一次（main.py 的 lifespan），全局复用；
M3 起的多租户 ORM 模型、CRUD 都共用它。

注意 create_async_engine 是"懒连接"的——创建时不真正连 DB，首次借连接才连。
所以即使 DB 没起，应用也能正常启动，由 /health 如实报告其状态（降级不阻断）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


def build_engine() -> AsyncEngine:
    """按配置创建异步引擎（内含连接池）。整个进程只调一次。"""
    return create_async_engine(
        settings.database_url,  # postgresql+asyncpg://...
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=settings.db_pool_pre_ping,
        echo=settings.db_echo,
    )


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """会话工厂。expire_on_commit=False：commit 后对象仍可读，
    避免异步环境里意外触发懒加载 IO。"""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def ping(engine: AsyncEngine) -> None:
    """探活：从池借一条连接执行 SELECT 1。"""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


def pool_stats(engine: AsyncEngine) -> dict[str, int]:
    """读出池的实时状态——这样 /health 能让你"看见"连接池在工作。"""
    pool = engine.pool
    return {
        "size": pool.size(),  # 配置的常驻连接数
        "checked_out": pool.checkedout(),  # 当前被借出（正在用）的连接数
        "checked_in": pool.checkedin(),  # 当前空闲在池中、可复用的连接数
    }


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：从 app.state 的会话工厂取一个 Session，请求结束自动关闭。
    M3 起的路由用 `session: AsyncSession = Depends(get_session)` 注入。"""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.db_sessionmaker
    async with sessionmaker() as session:
        yield session
