"""Alembic 迁移环境（异步版）。

要点：
  - 连接串从 app.core.config.settings 读（不在 .ini 里硬编码）——配置单一事实来源；
  - 用异步引擎（asyncpg）连库，但迁移本身是同步执行的，靠 connection.run_sync 桥接；
  - target_metadata = Base.metadata，且**必须 import 所有模型**，否则 autogenerate
    看不到表、会生成空迁移（经典坑）。
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import settings

# import 模型以填充 Base.metadata（autogenerate 据此对比"模型 vs 数据库"）
from app.db.models import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 把运行时的连接串塞进 alembic 配置（覆盖 .ini 里的空值）
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata

# LangGraph PostgresSaver 自己建的 checkpoint 表（M1），不归 Alembic 管。
# 不排除的话，autogenerate 会发现"库里有、模型里没有"，生成 DROP 把短期记忆删掉——
# 这是 autogenerate 的经典坑：它默认想把库收敛成 metadata 的样子。
_LANGGRAPH_TABLES = {
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    """autogenerate 过滤器：忽略不属于本应用 metadata 的 LangGraph 表。"""
    if type_ == "table" and name in _LANGGRAPH_TABLES:
        return False
    return True


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # 比较列类型变化，迁移更精确
        compare_type=True,
        include_object=_include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在线模式：建异步引擎 → 连接 → run_sync 桥接到同步的迁移执行。"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # 迁移是一次性短任务，不需要连接池
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 不连库（alembic upgrade --sql）。"""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
