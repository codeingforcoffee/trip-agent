"""短期记忆后端：基于 Postgres 的 LangGraph checkpointer。

checkpointer 是 LangGraph 的"状态持久化"插件——图每走完一步，就把整个 State
（消息历史等）按 thread_id 存一份快照到后端。于是：
  - 多轮对话：同一 thread_id 再次 ainvoke 时，框架自动把历史 State 读回来接着走，
    这就是"短期记忆"——不是模型记住了，而是状态被持久化并回放了；
  - 断点续跑：进程崩溃/重启后，用同一 thread_id 仍能恢复对话（M1 验证会演示）。

我们用 AsyncPostgresSaver（真实连接 Postgres，对应需求里的"真实连接数据库"）。
注意它底层用的是 psycopg3（不是我们 M0 给业务用的 asyncpg），所以连接串用
裸 DSN settings.database_dsn（postgresql://...，不带 +asyncpg）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings


@asynccontextmanager
async def open_checkpointer(setup: bool = True) -> AsyncIterator[AsyncPostgresSaver]:
    """打开一个 Postgres checkpointer（异步上下文管理器，单连接）。

    setup=True 时调用 .setup() 建表（幂等，首次会创建 checkpoints 等表）。
    这是 **CLI / 一次性脚本** 用的简单版：每进程一条连接足够。
    Web 服务（M9 的 FastAPI）并发请求多，改用下面 build_checkpointer_pool 的连接池版。
    """
    async with AsyncPostgresSaver.from_conn_string(settings.database_dsn) as checkpointer:
        if setup:
            await checkpointer.setup()
        yield checkpointer


def build_checkpointer_pool() -> AsyncConnectionPool:
    """构建 FastAPI 用的【长生命周期】Postgres 连接池（M9）。

    为什么 Web 服务必须用池、不能用 open_checkpointer 的单连接：
      psycopg 的一条连接**不能被多个协程并发共用**——两个请求同时命中同一连接会串包/报错。
      CLI 是单请求串行，一条连接够；Web 高并发下必须池化，每个借出的连接同一时刻只服务一个请求。

    两个 kwargs 是 AsyncPostgresSaver 的硬要求 / 稳健性考量：
      - autocommit=True：checkpointer 自己管事务边界，连接必须处于自动提交模式；
      - prepare_threshold=0：关掉预备语句缓存——建表/查询语句多变，且过 PgBouncer 之类
        连接池代理时预备语句易冲突，关掉最稳。

    open=False + 由调用方 await pool.open()：新版 psycopg_pool 不建议在构造函数里就打开池
    （构造期打开会告警），交给 lifespan 显式 open/close，生命周期更清晰。
    """
    return AsyncConnectionPool(
        conninfo=settings.database_dsn,
        min_size=1,
        max_size=settings.checkpointer_pool_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
