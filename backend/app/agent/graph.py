"""LangGraph 图的定义：agent ↔ tools 的 ReAct 循环。

把 M1 心智模型落成代码：
  START → agent → (有 tool_calls ? tools : END)
                    tools → agent  （回到上面，让模型看到工具结果继续）

设计说明：
  - 节点是闭包，把 llm/tools 通过 build_graph 参数注入，避免全局耦合、方便测试；
  - 身份/租户等"每次调用都不同、但不属于对话内容"的东西，走 LangGraph 的
    config.configurable（M3 会注入 tenant_id/user_id），而不是塞进 State；
  - M2 起 tools 节点用 asyncio.gather **并发**执行一轮里的多个 tool_call，
    单个工具的异常被隔离在自己的任务里（部分失败不连累其它），并回灌给模型自纠。
"""

from __future__ import annotations

import asyncio
from datetime import date
from time import perf_counter

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent import memory
from app.agent.context import latest_usage_tokens, make_llm_summarizer, maybe_compress
from app.agent.reliability import CircuitOpen, call_tool_resilient
from app.agent.state import AgentState
from app.agent.tools import ALL_TOOLS, TOOLS_BY_NAME
from app.agent.triage import (
    TRIAGE_SYSTEM_PROMPT,
    TripIntent,
    build_clarify_question,
    clarification_needs,
    route_after_triage,
)
from app.core.config import settings
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
5. 政策类问题（报销标准/流程/职级权限）必须以 query_travel_policy 返回的条款为准，并在答案中标注来源
   （如「依据《xx办法》住宿标准」）；若检索结果显示未找到，如实告知用户，不要编造金额或规则。
"""


def _system_message() -> SystemMessage:
    return SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat()))


def _summary_prefix(state: AgentState) -> list[AnyMessage]:
    """把滚动摘要(若有)拼成一条 SystemMessage，垫在历史之前。

    这是 M6a 压缩"不丢线索"的关键：被淘汰的旧消息已浓缩进 summary，agent/triage 读到它，
    就还能跨轮补槽、知道用户此前要干什么。无摘要时返回空列表（不影响早期对话）。
    """
    summary = state.get("summary")
    if not summary:
        return []
    return [SystemMessage(content=f"【对话摘要（更早历史已压缩，仅供参考）】\n{summary}")]


def _memory_prefix(state: AgentState) -> list[AnyMessage]:
    """把召回的长期记忆（偏好 + 相关历史）拼成一条 SystemMessage，供 agent 主动应用。"""
    mem = state.get("memory_context")
    if not mem:
        return []
    return [SystemMessage(content=f"【关于该用户的已知长期记忆，可在建议中主动应用】\n{mem}")]


def _last_human_text(messages: list[AnyMessage]) -> str:
    """取最后一条用户消息文本（作为语义召回的查询）。"""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content or "")
    return ""


def should_continue(state: AgentState) -> str:
    """条件边：看最后一条 AI 消息有没有要调的工具。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


async def _run_tool_call(
    call: dict, tools_by_name: dict[str, BaseTool], config: dict | None = None
) -> ToolMessage:
    """执行单个 tool_call，**永远**返回一条 ToolMessage（错误也包成消息，绝不抛出）。

    为什么 try/except 包在这一层而不是 gather 外面：tools 节点会并发跑多个工具，
    任一工具抛异常都不该连累其它工具——这就是「部分失败隔离」。被捕获的异常
    （含 pydantic 参数校验错误）作为文本回灌给模型，触发它修正参数后重试（自纠）。

    config（M5）：透传 LangGraph 的 RunnableConfig，让工具能从 config.configurable 读租户身份
    （如 RAG policy 的租户过滤）。模型看不到也改不了它——租户隔离的安全根基。

    抽成模块级函数（而非闭包）是为了能离线单测：直接喂一个 call dict 验证行为。
    """
    name = call["name"]
    tool = tools_by_name.get(name)
    if tool is None:
        # 模型幻觉出一个不存在的工具名时，告诉它而不是崩溃
        return ToolMessage(content=f"错误：未知工具 {name}", tool_call_id=call["id"], name=name)
    t0 = perf_counter()
    try:
        # M4：经可靠性封装调用——超时 + 重试(仅幂等) + 熔断。参数校验失败仍抛 ValidationError，
        # 被下面 except 兜住回灌给模型自纠（保持 M2 行为）。
        content = await call_tool_resilient(tool, call["args"], name=name, config=config)
        ok = True
    except CircuitOpen:
        # 熔断开路：优雅降级，告诉模型"暂不可用"，而非让它干等必然失败
        content = f"工具 {name} 暂时不可用（已熔断保护），请稍后再试或换个方式。"
        ok = False
    except Exception as e:  # noqa: BLE001 —— 故意兜底：任何异常都回灌给模型而非中断图
        content = f"工具 {name} 调用失败：{e}。请检查并修正参数后重试。"
        ok = False
    log.info(
        "tools.call_done",
        tool=name,
        ok=ok,
        elapsed_ms=round((perf_counter() - t0) * 1000, 1),
    )
    # ToolMessage 必须带 tool_call_id，框架据此把结果与那次调用配对
    return ToolMessage(content=content, tool_call_id=call["id"], name=name)


def build_graph(
    llm,
    tools: list[BaseTool] | None = None,
    checkpointer=None,
    enable_triage: bool | None = None,
    enable_compress: bool | None = None,
    enable_memory: bool | None = None,
) -> CompiledStateGraph:
    """组装并编译图。llm 为已构造的聊天模型；checkpointer 为短期记忆后端。

    enable_triage：是否启用"分诊+澄清"节点（M2+）。None 时取 settings.enable_triage。
    关掉就退回 M2 的 START→agent 直连图（省一次 LLM 调用，但不再主动澄清）。

    enable_compress：是否启用上下文压缩入口节点（M6a）。None 时取 settings.enable_compress。
    enable_memory：是否启用长期记忆（recall 入口 + memorize 收尾）（M6b）。None 时取 settings.enable_memory。
    """
    tools = tools if tools is not None else ALL_TOOLS
    tools_by_name = {t.name: t for t in tools} or TOOLS_BY_NAME
    if enable_triage is None:
        enable_triage = settings.enable_triage
    if enable_compress is None:
        enable_compress = settings.enable_compress
    if enable_memory is None:
        enable_memory = settings.enable_memory
    # bind_tools：把工具的 JSON Schema 告诉模型，模型才会产出 tool_calls
    llm_with_tools = llm.bind_tools(tools)
    # triage 用结构化输出，提前 bind 一次复用；关闭时不调用，便于无该能力的 LLM 直连
    triage_llm = llm.with_structured_output(TripIntent) if enable_triage else None
    # M6a：摘要器复用同一个 llm（只在超预算时才真正调用，平时 compress 是纯计数的 pass-through）
    summarizer = make_llm_summarizer(llm)

    async def agent_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """agent 节点：把系统提示 +（滚动摘要）+（长期记忆）+ 历史消息喂给模型，拿回它的决策/回答。"""
        messages = [
            _system_message(),
            *_summary_prefix(state),
            *_memory_prefix(state),
            *state["messages"],
        ]
        response = await llm_with_tools.ainvoke(messages)
        if response.tool_calls:
            log.info(
                "agent.decide_tools",
                calls=[c["name"] for c in response.tool_calls],
            )
        return {"messages": [response]}

    async def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, list[AnyMessage]]:
        """tools 节点：**并发**执行最后一条 AI 消息里的所有 tool_calls。

        asyncio.gather 同时发起所有工具调用并等它们全部完成；它**保持顺序**，
        返回的 ToolMessage 与 tool_calls 一一对应（顺序不能乱，否则 id 配对会错）。
        对比日志里每个工具的 elapsed_ms 之和 与 这里的 wall_ms：并发下 wall_ms ≈
        最慢的那个工具，而非各工具之和——这就是 fan-out 的收益。

        config（M5）：LangGraph 把运行时配置作为第二参数注入节点；这里把它透传给每个工具，
        让 RAG 等工具能读到 config.configurable.tenant_id 做租户过滤。
        """
        calls = state["messages"][-1].tool_calls
        t0 = perf_counter()
        log.info("tools.fanout_start", count=len(calls), tools=[c["name"] for c in calls])
        outputs = await asyncio.gather(*(_run_tool_call(c, tools_by_name, config) for c in calls))
        log.info(
            "tools.fanout_done",
            count=len(calls),
            wall_ms=round((perf_counter() - t0) * 1000, 1),
        )
        return {"messages": list(outputs)}

    async def triage_node(state: AgentState) -> dict:
        """分诊节点（M2+）：用结构化输出从**整段对话**抽意图+槽位，算出待澄清点。

        喂完整历史而非只喂最后一句，是为了天然处理多轮补槽/指代：第二轮"北京出发"
        时，triage 仍能从第一轮"我想去上海"里把 destination 抽回来，合并成完整槽位。
        """
        sys = SystemMessage(content=TRIAGE_SYSTEM_PROMPT.format(today=date.today().isoformat()))
        intent: TripIntent = await triage_llm.ainvoke(
            [sys, *_summary_prefix(state), *state["messages"]]
        )
        slots = {
            "origin": intent.origin,
            "destination": intent.destination,
            "date": intent.date,
        }
        needs = clarification_needs(intent.intent, slots)
        log.info("triage", intent=intent.intent, slots=slots, clarify_needs=needs)
        return {"intent": intent.intent, "slots": slots, "clarify_needs": needs}

    def clarify_node(state: AgentState) -> dict[str, list[AnyMessage]]:
        """澄清节点（M2+）：把待澄清点拼成一句反问返回，本轮到此结束、等用户补充。

        纯函数构造问题（不调 LLM）→ 确定性、零成本、可单测。它产出的是普通 AIMessage
        （无 tool_calls），所以 clarify→END，CLI 会把它当作助手发言打印出来。
        """
        question = build_clarify_question(state.get("clarify_needs", []))
        log.info("clarify.ask", needs=state.get("clarify_needs"))
        return {"messages": [AIMessage(content=question)]}

    async def compress_node(state: AgentState) -> dict:
        """压缩节点（M6a/M6+）：每轮入口先过它；用量超 high 水位才把旧轮次摘要 + 移除到 low 水位。

        平时是廉价的"读用量 → 没超 → pass-through"，只有超水位时才花一次 LLM 摘要。
        当前占用优先用上一轮真实 token 用量（含前缀），拿不到才回退字符估算。
        返回 RemoveMessage（add_messages reducer 据此删旧消息）+ 新 summary。
        """
        result = await maybe_compress(
            state["messages"],
            state.get("summary", ""),
            window_tokens=settings.context_window_tokens,
            high_ratio=settings.compress_high_ratio,
            low_ratio=settings.compress_low_ratio,
            keep_last_floor=settings.compress_keep_last,
            summarize=summarizer,
            used_tokens=latest_usage_tokens(state["messages"]),
        )
        if result is None:
            return {}
        log.info(
            "context.compress",
            evicted=result.evicted_count,
            kept=len(state["messages"]) - result.evicted_count,
            summary_chars=len(result.summary),
        )
        # 删旧消息 + 写回滚动摘要；二者在同一更新里提交，下游节点立即看到压缩后的状态
        return {
            "summary": result.summary,
            "messages": [RemoveMessage(id=mid) for mid in result.removed_ids],
        }

    async def recall_node(state: AgentState, config: RunnableConfig) -> dict:
        """召回节点（M6b）：每轮入口，把该用户的偏好 + 相关历史记忆注入 state.memory_context。

        无身份（离线结构测试不带 config）或召回失败 → 返回 {}（记忆是增强，不该中断对话）。
        """
        cfg = config.get("configurable") or {}
        tenant_id, user_id = cfg.get("tenant_id"), cfg.get("user_id")
        if not tenant_id or not user_id:
            return {}
        try:
            ctx = await memory.recall(
                _last_human_text(state["messages"]), tenant_id=str(tenant_id), user_id=str(user_id)
            )
        except Exception as e:  # noqa: BLE001
            log.warning("memory.recall_failed", error=repr(e))
            return {}
        if not ctx:
            return {}
        log.info("memory.recall", chars=len(ctx))
        return {"memory_context": ctx}

    async def memorize_node(state: AgentState, config: RunnableConfig) -> dict:
        """收尾节点（M6b）：turn 结束抽取值得记的偏好/事实并写入长期记忆。

        当前**同步**实现（简单、CLI 可复现）；生产可改 write-behind（后台队列）不阻塞用户响应。
        写入失败不中断对话。
        """
        cfg = config.get("configurable") or {}
        tenant_id, user_id = cfg.get("tenant_id"), cfg.get("user_id")
        if not tenant_id or not user_id:
            return {}
        try:
            stats = await memory.memorize(
                llm, state["messages"], tenant_id=str(tenant_id), user_id=str(user_id)
            )
            if stats:
                log.info("memory.write", **stats)
        except Exception as e:  # noqa: BLE001
            log.warning("memory.write_failed", error=repr(e))
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge("tools", "agent")  # 工具跑完回到 agent，形成 ReAct 循环

    # agent 之后：还要调工具 → tools；否则去"收尾"。开了长期记忆时收尾是 memorize，否则直接 END。
    end_target = END
    if enable_memory:
        builder.add_node("memorize", memorize_node)
        builder.add_edge("memorize", END)
        end_target = "memorize"
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": end_target})

    # triage/clarify（M2+）
    core_entry = "agent"
    if enable_triage:
        builder.add_node("triage", triage_node)
        builder.add_node("clarify", clarify_node)
        builder.add_conditional_edges(
            "triage", route_after_triage, {"clarify": "clarify", "agent": "agent"}
        )
        # 澄清轮也走收尾：用户可能在"信息不全"的一句里顺带表达了偏好（如"记住我只坐靠窗"被
        # 误判成缺槽的 flight），仍要能抽取记忆，而不是被 clarify→END 吞掉。
        builder.add_edge("clarify", end_target)
        core_entry = "triage"

    # 入口链：START →(compress M6a)→(recall M6b)→ core_entry。按开关拼接，避免各组合写死。
    chain: list[str] = []
    if enable_compress:
        builder.add_node("compress", compress_node)
        chain.append("compress")
    if enable_memory:
        builder.add_node("recall", recall_node)
        chain.append("recall")
    chain.append(core_entry)
    builder.add_edge(START, chain[0])
    for a, b in zip(chain, chain[1:], strict=False):
        builder.add_edge(a, b)

    # compile 时传入 checkpointer：图每走一步都会把 State 存进它（=短期记忆）
    return builder.compile(checkpointer=checkpointer)
