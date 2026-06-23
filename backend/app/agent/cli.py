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


async def run_turn(graph, config: dict, text: str) -> None:
    """跑一轮对话，流式打印每个节点的中间过程。"""
    print(f"\n🧑 你> {text}")
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=text)]}, config, stream_mode="updates"
    ):
        for node, update in chunk.items():
            _print_node_messages(node, update.get("messages", []))


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


async def run_once(thread_id: str, message: str) -> None:
    async with open_checkpointer() as cp:
        graph = build_graph(get_llm(), ALL_TOOLS, checkpointer=cp)
        await run_turn(graph, {"configurable": {"thread_id": thread_id}}, message)


async def run_interactive(thread_id: str) -> None:
    print(f"进入交互模式（thread_id={thread_id}）。输入 exit/quit 退出。")
    async with open_checkpointer() as cp:
        graph = build_graph(get_llm(), ALL_TOOLS, checkpointer=cp)
        config = {"configurable": {"thread_id": thread_id}}
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
    parser.add_argument("--message", help="一次性发送一条消息后退出")
    parser.add_argument("--history", action="store_true", help="打印该会话的历史消息")
    args = parser.parse_args()

    if args.history:
        asyncio.run(show_history(args.thread))
    elif args.message:
        asyncio.run(run_once(args.thread, args.message))
    else:
        asyncio.run(run_interactive(args.thread))


if __name__ == "__main__":
    main()
