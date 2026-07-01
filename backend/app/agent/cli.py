"""差旅 Agent 命令行入口（M1）。

用法（在 backend/ 下）：
  uv run python -m app.agent.cli --message "明天北京飞上海"          # 一次性问一句
  uv run python -m app.agent.cli --thread alice                      # 进入交互模式
  uv run python -m app.agent.cli --thread alice --history            # 查看某会话的历史

每次带 --message 运行都是一个**独立进程**，但只要 thread_id 相同，对话历史就会
从 Postgres 恢复——这正是"短期记忆 = 状态持久化"和"断点续跑"的现场证明。
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from app.agent.checkpointer import open_checkpointer
from app.agent.graph import build_graph
from app.agent.tools import ALL_TOOLS
from app.core.logging import setup_logging
from app.llm.deepseek import get_llm


def _truncate(s: str, n: int = 200) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _print_node_messages(node: str, messages: list) -> None:
    """把图每一步的输出渲染成人类可读的过程，方便看清 ReAct 循环。"""
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls or []:
                print(f"  🤔 [agent] 决定调用工具 {call['name']}({call['args']})")
            if m.content:
                print(f"\n🤖 助手> {m.content}")
        elif isinstance(m, ToolMessage):
            print(f"  🔧 [tools] {m.name} 返回: {_truncate(str(m.content))}")


async def _ask_confirm(interrupts) -> dict:
    """收到 HITL 中断时，向用户展示待确认的高危动作并读取批准/拒绝。

    interrupts 是 __interrupt__ chunk 的值：一个 Interrupt 元组，.value 是 confirm 节点抛出的 payload。
    """
    payload = interrupts[0].value if interrupts else {}
    print(f"\n⚠️  {payload.get('message', '需要确认高危操作：')}")
    for a in payload.get("actions", []):
        print(f"    - {a['tool']}({a['args']})")
    loop = asyncio.get_running_loop()
    ans = await loop.run_in_executor(None, input, "  批准执行? [y/N] > ")
    approved = ans.strip().lower() in {"y", "yes", "是", "确认"}
    print(f"  → {'✅ 已批准' if approved else '⛔ 已拒绝'}")
    return {"approved": approved}


async def run_turn(graph, config: dict, text: str) -> None:
    """跑一轮对话，流式打印每个节点的中间过程；遇到 HITL 中断则询问用户后 resume。

    M7 起图可能在 confirm 节点 interrupt 暂停：astream 会吐一个 {"__interrupt__": (...)} 后结束，
    此时用 Command(resume=decision) 重新 astream 恢复，直到不再中断（while 循环）。
    """
    print(f"\n🧑 你> {text}")
    step_input = {"messages": [HumanMessage(content=text)]}
    while step_input is not None:
        resume_with = None
        async for chunk in graph.astream(step_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    resume_with = Command(resume=await _ask_confirm(update))
                else:
                    # updates 模式下，无状态更新的节点（如未触发压缩的 compress）其 update 为 None，兜一下
                    _print_node_messages(node, (update or {}).get("messages", []))
        step_input = resume_with  # 有中断则带确认结果再跑一轮，否则结束


async def show_history(thread_id: str) -> None:
    async with open_checkpointer(setup=False) as cp:
        graph = build_graph(get_llm(), ALL_TOOLS, checkpointer=cp)
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        msgs = state.values.get("messages", []) if state and state.values else []
        print(f"== 会话 thread_id={thread_id} 共有 {len(msgs)} 条消息 ==")
        for m in msgs:
            role = {
                HumanMessage: "用户",
                AIMessage: "助手",
                ToolMessage: "工具",
                SystemMessage: "系统",
            }.get(type(m), type(m).__name__)
            extra = ""
            if isinstance(m, AIMessage) and m.tool_calls:
                extra = f" [调用: {[c['name'] for c in m.tool_calls]}]"
            print(f"  [{role}] {_truncate(str(m.content) or '(空)')}{extra}")


async def _resolve_ids(slug: str, email: str | None) -> tuple[str | None, str | None, list[str]]:
    """按 slug 查 tenant_id、按 email(或该租户首个用户)查 user_id + scopes，注入 config。

    tenant_id 供 RAG 租户过滤；user_id 供 M6b 长期记忆；**scopes 供 M7 工具授权**
    （高危工具据此放行/拒绝）。这里模拟"登录后从 JWT 拿到的可信身份"——真实链路里
    scopes 来自签名过的 token（core/security.py），CLI 直接查库取。超级用户连接（管理操作）。
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.core.config import settings
    from app.db.models import Tenant, User

    engine = create_async_engine(settings.database_url)
    try:
        async with engine.connect() as conn:
            trow = (await conn.execute(select(Tenant.id).where(Tenant.slug == slug))).first()
            if trow is None:
                return None, None, []
            tenant_id = trow[0]
            q = select(User.id, User.scopes).where(User.tenant_id == tenant_id)
            q = q.where(User.email == email) if email else q.order_by(User.created_at)
            urow = (await conn.execute(q.limit(1))).first()
            if urow is None:
                return str(tenant_id), None, []
            return str(tenant_id), str(urow[0]), list(urow[1] or [])
    finally:
        await engine.dispose()


def _config(
    thread_id: str, tenant_id: str | None, user_id: str | None, scopes: list[str] | None = None
) -> dict:
    """CLI 用的 RunnableConfig：thread_id 定短期记忆命名空间；tenant_id/user_id 供 RAG 与长期记忆；
    scopes 供 M7 工具授权。身份放 configurable（可信层），模型碰不到。"""
    return {
        "configurable": {
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "scopes": scopes or [],
        }
    }


async def run_once(thread_id: str, message: str, tenant: str, user: str | None) -> None:
    async with open_checkpointer() as cp:
        graph = build_graph(get_llm(), ALL_TOOLS, checkpointer=cp)
        tenant_id, user_id, scopes = await _resolve_ids(tenant, user)
        await run_turn(graph, _config(thread_id, tenant_id, user_id, scopes), message)


async def run_interactive(thread_id: str, tenant: str, user: str | None) -> None:
    tenant_id, user_id, scopes = await _resolve_ids(tenant, user)
    print(
        f"进入交互模式（thread_id={thread_id}，租户={tenant}，user={user_id}，"
        f"scopes={scopes}）。输入 exit/quit 退出。"
    )
    if tenant_id is None or user_id is None:
        print("⚠️  租户/用户未找到（先 make seed），政策检索与长期记忆将不可用。")
    async with open_checkpointer() as cp:
        graph = build_graph(get_llm(), ALL_TOOLS, checkpointer=cp)
        config = _config(thread_id, tenant_id, user_id, scopes)
        loop = asyncio.get_running_loop()
        while True:
            # input() 是阻塞调用，放进线程池避免卡住事件循环
            text = await loop.run_in_executor(None, input, "\n🧑 你> ")
            if text.strip().lower() in {"exit", "quit", ":q"}:
                print("再见 👋")
                break
            if not text.strip():
                continue
            await run_turn(graph, config, text)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="差旅 Agent CLI（M1）")
    parser.add_argument("--thread", default="demo", help="会话 ID（短期记忆按它隔离）")
    parser.add_argument("--tenant", default="acme", help="租户 slug（RAG 按租户隔离检索）")
    parser.add_argument(
        "--user", default=None, help="用户 email（长期记忆按用户隔离；缺省取该租户首个用户）"
    )
    parser.add_argument("--message", help="一次性发送一条消息后退出")
    parser.add_argument("--history", action="store_true", help="打印该会话的历史消息")
    args = parser.parse_args()

    if args.history:
        asyncio.run(show_history(args.thread))
    elif args.message:
        asyncio.run(run_once(args.thread, args.message, args.tenant, args.user))
    else:
        asyncio.run(run_interactive(args.thread, args.tenant, args.user))


if __name__ == "__main__":
    main()
