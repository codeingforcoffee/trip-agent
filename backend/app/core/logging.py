"""结构化日志（structlog）。

为什么从第一天就上 JSON 结构化日志，而不是 print / 普通 logging？
  - 企业里日志要被采集进 ELK / Loki，机器要能按字段检索（tenant_id、trace_id…）；
  - Agent 系统是异步 + 多租户的，没有结构化字段根本无法定位"哪个租户的哪次请求出了问题"；
  - contextvars 让我们在请求入口 bind 一次身份，后续所有日志自动带上，无需层层传参。

用法：
    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("tool.call", tool="flights", duration_ms=42)

绑定上下文（M3 的中间件会做）：
    structlog.contextvars.bind_contextvars(trace_id=..., tenant_id=..., user_id=...)
"""

from __future__ import annotations

import logging

import structlog

from app.core.config import settings


def setup_logging() -> None:
    """进程启动时调用一次，配置 structlog 全局处理链。"""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # 处理链：每条日志依次经过这些 processor
    processors: list = [
        # 把 contextvars 里 bind 的字段（trace_id/tenant_id…）合并进每条日志
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,  # 加 level 字段
        structlog.processors.TimeStamper(fmt="iso"),  # 加 ISO 时间戳
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,  # 异常渲染成可读堆栈
    ]

    if settings.log_json:
        # 生产：一行一个 JSON。ensure_ascii=False 让中文日志直接可读，不转义成 \uXXXX
        processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))
    else:
        # 本地开发：带颜色的人类可读输出
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        # 低于 log_level 的日志直接丢弃，零开销
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
