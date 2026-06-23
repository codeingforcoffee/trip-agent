"""LangGraph 图的定义：agent ↔ tools 的 ReAct 循环。

把 M1 心智模型落成代码：
  START → agent → (有 tool_calls ? tools : END)
                    tools → agent  （回到上面，让模型看到工具结果继续）

设计说明：
  - 节点是闭包，把 llm/tools 通过 build_graph 参数注入，避免全局耦合、方便测试；
  - 身份/租户等"每次调用都不同、但不属于对话内容"的东西，走 LangGraph 的
    config.configurable（M3 会注入 tenant_id/user_id），而不是塞进 State；
  - M1 的 tools 节点是**顺序**执行的（一次只跑一个工具调用），保持透明易懂；
    M2 会把它升级为 asyncio.gather 并发执行 + 错误自纠。
"""

from __future__ import annotations

from datetime import date

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.state import AgentState
from app.agent.tools import ALL_TOOLS, TOOLS_BY_NAME
from app.core.logging import get_logger

log = get_logger("app.agent.graph")

# 系统提示：定义 Agent 的角色与行为准则。{today} 在运行时填入，
# 这样模型能把"明天/后天"换算成具体日期再调工具。
SYSTEM_PROMPT = """你是一个专业、严谨的企业差旅助手。今天的日期是 {today}。

工作准则：
1. 需要实时信息（航班、酒店、天气、报销政策等）时，必须调用相应工具获取，不要凭空编造。
2. 用户说"明天/后天/下周一"等相对日期时，先根据"今天日期"换算成 YYYY-MM-DD 再传给工具。
3. 信息不足以调用工具时（如缺出发城市），先用一句话向用户追问，不要瞎猜。
4. 回答简洁、结构化，金额用人民币元，给出可执行的建议（如推荐最具性价比的航班）。
"""


def _system_message() -> SystemMessage:
    return SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat()))


def should_continue(state: AgentState) -> str:
    """条件边：看最后一条 AI 消息有没有要调的工具。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def build_graph(
    llm,
    tools: list[BaseTool] | None = None,
    checkpointer=None,
) -> CompiledStateGraph:
    """组装并编译图。llm 为已构造的聊天模型；checkpointer 为短期记忆后端。"""
    tools = tools if tools is not None else ALL_TOOLS
    tools_by_name = {t.name: t for t in tools} or TOOLS_BY_NAME
    # bind_tools：把工具的 JSON Schema 告诉模型，模型才会产出 tool_calls
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """agent 节点：把系统提示 + 历史消息喂给模型，拿回它的决策/回答。"""
        messages = [_system_message(), *state["messages"]]
        response = await llm_with_tools.ainvoke(messages)
        if response.tool_calls:
            log.info(
                "agent.decide_tools",
                calls=[c["name"] for c in response.tool_calls],
            )
        return {"messages": [response]}

    async def tools_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """tools 节点：执行最后一条 AI 消息里的所有 tool_calls（M1 顺序执行）。"""
        last = state["messages"][-1]
        outputs: list[AnyMessage] = []
        for call in last.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                content = f"错误：未知工具 {call['name']}"
            else:
                try:
                    # ainvoke 接收参数 dict；同步工具会被自动放到线程池里跑
                    content = str(await tool.ainvoke(call["args"]))
                except Exception as e:  # noqa: BLE001 —— 工具异常要回灌给模型让它自纠，而非崩溃
                    content = f"工具 {call['name']} 执行出错：{e!r}"
            log.info("tools.executed", tool=call["name"], args=call["args"])
            # ToolMessage 必须带 tool_call_id，框架据此把结果和那次调用配对
            outputs.append(
                ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
            )
        return {"messages": outputs}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    # 条件边：agent 之后，按 should_continue 的返回值选择去向
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")  # 工具跑完回到 agent，形成循环
    # compile 时传入 checkpointer：图每走一步都会把 State 存进它（=短期记忆）
    return builder.compile(checkpointer=checkpointer)
