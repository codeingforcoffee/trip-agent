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

from app.core.config import settings


@asynccontextmanager
async def open_checkpointer(setup: bool = True) -> AsyncIterator[AsyncPostgresSaver]:
    """打开一个 Postgres checkpointer（异步上下文管理器）。

    setup=True 时调用 .setup() 建表（幂等，首次会创建 checkpoints 等表）。
    M9 接入 FastAPI 时会改用长连接池版本；CLI/M1 用这个简单连接即可。
    """
    async with AsyncPostgresSaver.from_conn_string(settings.database_dsn) as checkpointer:
        if setup:
            await checkpointer.setup()
        yield checkpointer
