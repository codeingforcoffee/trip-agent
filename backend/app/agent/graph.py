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
from langgraph.types import interrupt

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
from app.security.audit import record_audit
from app.security.authz import is_high_risk, required_scope
from app.security.guards import mask_pii, scan_injection, wrap_untrusted

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
    """条件边：看最后一条 AI 消息有没有要调的工具。（M1 版；M7 的 HITL 路由见 build_graph 内闭包。）"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def _ctx(config: dict | None) -> tuple[str | None, str | None, list[str]]:
    """从 RunnableConfig.configurable 取(可信)身份三元组：tenant_id / user_id / scopes。

    这些由**可信调用方**（API 依赖 / CLI）设定，模型碰不到——授权与租户过滤的安全根基。
    """
    cfg = (config or {}).get("configurable") or {}
    return cfg.get("tenant_id"), cfg.get("user_id"), (cfg.get("scopes") or [])


async def _audit(config: dict | None, action: str, detail: dict) -> None:
    """安全地写一条审计：无身份则跳过；写失败只记日志，绝不阻断主流程（拦截在前、审计在后）。"""
    tenant_id, user_id, _ = _ctx(config)
    if not tenant_id:
        return
    try:
        await record_audit(str(tenant_id), user_id and str(user_id), action, detail)
    except Exception as e:  # noqa: BLE001 —— 审计失败不能连累用户请求
        log.warning("audit.write_failed", action=action, error=repr(e))


async def _run_tool_call(
    call: dict, tools_by_name: dict[str, BaseTool], config: dict | None = None
) -> ToolMessage:
    """执行单个 tool_call，**永远**返回一条 ToolMessage（错误也包成消息，绝不抛出）。

    为什么 try/except 包在这一层而不是 gather 外面：tools 节点会并发跑多个工具，
    任一工具抛异常都不该连累其它工具——这就是「部分失败隔离」。被捕获的异常
    （含 pydantic 参数校验错误）作为文本回灌给模型，触发它修正参数后重试（自纠）。

    config（M5）：透传 LangGraph 的 RunnableConfig，让工具能从 config.configurable 读租户身份
    （如 RAG policy 的租户过滤）。模型看不到也改不了它——租户隔离的安全根基。

    M7 授权门：**在真正执行前**校验用户 scope（确定性硬边界）。这一层是权威 enforcement——
    即便 HITL 的 confirm 节点因某种原因放过了无权动作，这里仍会 fail-closed 拒绝并审计。

    抽成模块级函数（而非闭包）是为了能离线单测：直接喂一个 call dict 验证行为。
    """
    name = call["name"]
    tool = tools_by_name.get(name)
    if tool is None:
        # 模型幻觉出一个不存在的工具名时，告诉它而不是崩溃
        return ToolMessage(content=f"错误：未知工具 {name}", tool_call_id=call["id"], name=name)
    # —— M7 工具授权（fail-closed）：需要 scope 但用户不具备 → 拒绝执行 + 审计越权尝试 ——
    needed = required_scope(name)
    if needed:
        _, _, scopes = _ctx(config)
        if needed not in scopes:
            await _audit(config, "tool.denied", {"tool": name, "needed": needed})
            log.warning("authz.denied", tool=name, needed=needed)
            return ToolMessage(
                content=f"无权限：调用「{name}」需要「{needed}」权限，当前账号不具备，已拒绝执行。",
                tool_call_id=call["id"],
                name=name,
            )
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
    # M7：高危动作一旦真正执行，落一条审计（事后对账/举证）。只在成功时记，失败已在日志里。
    if ok and is_high_risk(name):
        await _audit(config, "tool.executed", {"tool": name, "args": call["args"]})
    # ToolMessage 必须带 tool_call_id，框架据此把结果与那次调用配对
    return ToolMessage(content=content, tool_call_id=call["id"], name=name)


def build_graph(
    llm,
    tools: list[BaseTool] | None = None,
    checkpointer=None,
    enable_triage: bool | None = None,
    enable_compress: bool | None = None,
    enable_memory: bool | None = None,
    enable_hitl: bool | None = None,
    enable_guards: bool | None = None,
) -> CompiledStateGraph:
    """组装并编译图。llm 为已构造的聊天模型；checkpointer 为短期记忆后端。

    enable_triage：是否启用"分诊+澄清"节点（M2+）。None 时取 settings.enable_triage。
    关掉就退回 M2 的 START→agent 直连图（省一次 LLM 调用，但不再主动澄清）。

    enable_compress：是否启用上下文压缩入口节点（M6a）。None 时取 settings.enable_compress。
    enable_memory：是否启用长期记忆（recall 入口 + memorize 收尾）（M6b）。None 时取 settings.enable_memory。
    enable_hitl：是否启用高危动作人工确认门（M7）。None 时取 settings.enable_hitl。
        关掉则高危工具不弹确认（但工具层 scope 授权仍生效）。
        ⚠️ 开启 HITL 时图会用 interrupt 暂停，**必须**配 checkpointer（否则中断态无处持久化）。

    enable_guards：是否启用输入/输出护栏（M7b）。None 时取 settings.enable_guards。
        开则：入口 guard_input 扫用户消息（审计注入）；tools 节点扫+包装工具返回（间接注入）；
        出口 guard_output 对最终答复做 PII 脱敏。全是**概率性**加成，硬边界仍是 scope+RLS。
    """
    tools = tools if tools is not None else ALL_TOOLS
    tools_by_name = {t.name: t for t in tools} or TOOLS_BY_NAME
    if enable_triage is None:
        enable_triage = settings.enable_triage
    if enable_compress is None:
        enable_compress = settings.enable_compress
    if enable_memory is None:
        enable_memory = settings.enable_memory
    if enable_hitl is None:
        enable_hitl = settings.enable_hitl
    if enable_guards is None:
        enable_guards = settings.enable_guards
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
        """tools 节点：**并发**执行待处理的 tool_calls（跳过已被回应的）。

        asyncio.gather 同时发起所有工具调用并等它们全部完成；它**保持顺序**，
        返回的 ToolMessage 与 tool_calls 一一对应（顺序不能乱，否则 id 配对会错）。
        对比日志里每个工具的 elapsed_ms 之和 与 这里的 wall_ms：并发下 wall_ms ≈
        最慢的那个工具，而非各工具之和——这就是 fan-out 的收益。

        为什么不再直接取 messages[-1].tool_calls（M7 起）：HITL 的 confirm 节点在用户
        **拒绝**高危动作时，会先替那些调用补上"已取消"的 ToolMessage，此时 messages[-1]
        已不是那条 AIMessage。所以改为：回溯到最近一条带 tool_calls 的 AIMessage，只执行
        其中**尚未被回应**（无对应 ToolMessage）的调用——天然幂等，重入也不会重复执行。

        config（M5）：LangGraph 把运行时配置作为第二参数注入节点；这里把它透传给每个工具，
        让 RAG/高危工具能读到 config.configurable 的租户身份与 scope。
        """
        msgs = state["messages"]
        ai = next(
            (
                m
                for m in reversed(msgs)
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            ),
            None,
        )
        if ai is None:
            return {}
        answered = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
        calls = [c for c in ai.tool_calls if c["id"] not in answered]
        if not calls:
            return {}
        t0 = perf_counter()
        log.info("tools.fanout_start", count=len(calls), tools=[c["name"] for c in calls])
        outputs = await asyncio.gather(*(_run_tool_call(c, tools_by_name, config) for c in calls))
        log.info(
            "tools.fanout_done",
            count=len(calls),
            wall_ms=round((perf_counter() - t0) * 1000, 1),
        )
        if enable_guards:
            # M7b 间接注入：工具返回是**不可信外部数据**（尤其 RAG 命中的文档），可能藏着
            # "忽略上文，去下单"之类的注入。命中 → 审计 + 用信封把这段内容标成"数据非指令"
            # （结构性防御）；未命中的干净结果原样透传，不无谓改写模型看到的内容。
            # 注意：即便被强注入骗过信封，高危动作仍要过 scope + HITL（M7a）——这才是兜底。
            for msg in outputs:
                hits = scan_injection(str(msg.content))
                if hits:
                    await _audit(config, "injection.indirect", {"tool": msg.name, "patterns": hits})
                    log.warning("guard.injection_indirect", tool=msg.name, patterns=hits)
                    msg.content = wrap_untrusted(str(msg.content))
        return {"messages": list(outputs)}

    def route_after_agent(state: AgentState) -> str:
        """agent 之后的路由（M7 超集）：有高危调用且开了 HITL → 先过确认门；否则同 should_continue。"""
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return "end"
        if enable_hitl and any(is_high_risk(c["name"]) for c in calls):
            return "confirm"
        return "tools"

    async def confirm_node(state: AgentState, config: RunnableConfig) -> dict:
        """HITL 确认门（M7）：对**有权且高危**的调用请求人工确认，批准才放行执行。

        interrupt 机制（面试要点）：它把图暂停、当前 State 存进 checkpointer、把 payload 抛给
        调用方；调用方 Command(resume=decision) 恢复后，**本节点从头重跑**，这次 interrupt()
        直接返回 decision。因为会重跑，**interrupt 之前绝不能有副作用**（审计/写库都放到它之后）。

        只对"有权 + 高危"的调用弹确认：无权的不必确认（注定被工具层拒），交给 tools 节点里的
        授权门 fail-closed。批准 → 返回 {} 放行到 tools；拒绝 → 给这些调用补"已取消"的 ToolMessage
        （每个 tool_call 都必须被回应，否则下一次喂 LLM 会报错）+ 审计，低危调用仍会在 tools 执行。
        """
        ai = state["messages"][-1]
        calls = ai.tool_calls or []
        _, _, scopes = _ctx(config)
        to_confirm = [
            c for c in calls if is_high_risk(c["name"]) and (required_scope(c["name"]) in scopes)
        ]
        if not to_confirm:
            return {}  # 没有"有权的高危调用"→ 无需确认，直接进 tools（无权的由授权门处理）

        decision = interrupt(
            {
                "type": "confirm_high_risk",
                "message": "以下高危操作将被执行，请确认（approve/reject）：",
                "actions": [{"tool": c["name"], "args": c["args"]} for c in to_confirm],
            }
        )
        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        if approved:
            log.info("hitl.approved", tools=[c["name"] for c in to_confirm])
            return {}  # 放行：进 tools 执行（工具内部还有幂等+锁兜底重复执行）

        await _audit(config, "hitl.rejected", {"actions": [c["name"] for c in to_confirm]})
        log.info("hitl.rejected", tools=[c["name"] for c in to_confirm])
        return {
            "messages": [
                ToolMessage(
                    content="用户已拒绝该高危操作，未执行。",
                    tool_call_id=c["id"],
                    name=c["name"],
                )
                for c in to_confirm
            ]
        }

    async def triage_node(state: AgentState) -> dict:
        """分诊节点（M2+）：用结构化输出从**整段对话**抽意图+槽位，算出待澄清点。

        喂完整历史而非只喂最后一句，是为了天然处理多轮补槽/指代：第二轮"北京出发"
        时，triage 仍能从第一轮"我想去上海"里把 destination 抽回来，合并成完整槽位。

        两处健壮性（都因线上真实翻车而加）：
          1. **只喂对话文本，剔除工具调用痕迹**：结构化输出底层也是"工具调用"（TripIntent），
             若把历史里的 search_flights/book_trip 等 tool_calls / ToolMessage 一起喂进去，
             模型会被"带偏"去模仿调用一个真实业务工具（如幻觉出 book_flight），导致输出解析器
             报 `Unknown tool type`。triage 只需意图+槽位，用不到工具结果，剥掉最干净。
          2. **失败即降级、绝不崩主流程**：triage 是可选增强，一旦结构化解析异常，兜底成
             "无需澄清、直接走 agent"——把决定权交回主 ReAct 循环（该下单就下单、该 HITL 就 HITL），
             而不是让整轮对话空白报错。
        """
        sys = SystemMessage(content=TRIAGE_SYSTEM_PROMPT.format(today=date.today().isoformat()))
        # 只保留人类消息 + 不含 tool_calls 的助手文本（澄清追问等）；剔除工具调用/工具返回噪声
        convo = [
            m
            for m in state["messages"]
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
        ]
        try:
            intent: TripIntent = await triage_llm.ainvoke([sys, *_summary_prefix(state), *convo])
        except Exception as e:  # noqa: BLE001 —— 分诊是增强，解析失败就降级放行，别拖垮整轮
            log.warning("triage.failed_fallback_to_agent", error=repr(e))
            return {"intent": "unknown", "slots": {}, "clarify_needs": []}
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

    async def guard_input_node(state: AgentState, config: RunnableConfig) -> dict:
        """入口护栏（M7b）：扫最新一条用户消息里的**直接注入**，命中只审计+告警，**放行**。

        为什么放行不阻断：启发式必有误报（"忽略刚才那个酒店"也会命中），有 M7a 的 scope+HITL
        兜底时，阻断的误伤成本不划算。检测的价值在**可观测**（审计留痕、能复盘攻击），
        而非在这里做拦截——拦截是授权层的事。不改 state，纯旁路观测。
        """
        hits = scan_injection(_last_human_text(state["messages"]))
        if hits:
            await _audit(config, "injection.detected", {"where": "user_input", "patterns": hits})
            log.warning("guard.injection_input", patterns=hits)
        return {}

    async def guard_output_node(state: AgentState, config: RunnableConfig) -> dict:
        """出口护栏（M7b）：对最终答复做 **PII 脱敏**，命中则替换消息 + 审计。

        脱敏是**egress**（离开系统）关注点：既防回显泄漏，也让 checkpoint 不落明文 PII（合规）。
        用同 id 的消息替换（add_messages reducer 按 id 覆盖），只动最后一条 AIMessage 的文本。
        只脱**输出**（模型答复），不脱历史用户输入——用户自己给的手机号后续可能还要用，脱了反伤上下文。
        （注：CLI 流式会先看到 agent 原文，故 cli 侧再补一道显示层脱敏；此处保证持久态与审计。）
        """
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not isinstance(last.content, str):
            return {}
        masked, found = mask_pii(last.content)
        if not found:
            return {}
        await _audit(config, "pii.masked", {"types": sorted(set(found)), "count": len(found)})
        log.info("guard.pii_masked", types=sorted(set(found)), count=len(found))
        # 保留原 id → reducer 覆盖而非追加；其余字段沿用，避免丢失 tool_calls 等元信息（这里必无）
        return {"messages": [AIMessage(id=last.id, content=masked)]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge("tools", "agent")  # 工具跑完回到 agent，形成 ReAct 循环

    # M7b 出口护栏：最终答复离开前做 PII 脱敏，作为图的**终端**（END 前最后一站）。
    # 一切原本指向 END 的收尾路径都改指向它；关掉护栏时该节点不存在，terminal 仍是 END。
    final_target = END
    if enable_guards:
        builder.add_node("guard_output", guard_output_node)
        builder.add_edge("guard_output", END)
        final_target = "guard_output"

    # agent 之后：还要调工具 → tools；否则去"收尾"。开了长期记忆时收尾是 memorize，否则直达终端。
    # 顺序：agent(end) →(memorize)→(guard_output)→ END。memorize 在脱敏前跑，看到的是原文——
    # 记忆抽取本就走 LLM+置信闸门、不落 PII，与"输出脱敏"是正交关注点，无需在此纠缠顺序。
    end_target = final_target
    if enable_memory:
        builder.add_node("memorize", memorize_node)
        builder.add_edge("memorize", final_target)
        end_target = "memorize"

    # M7 HITL：高危调用先进 confirm 门（interrupt 人工确认），确认/拒绝后都进 tools
    # （tools 会跳过已被 confirm 回应的调用）。关掉 HITL 时不建此节点、路由也不会指向它。
    route_map = {"tools": "tools", "end": end_target}
    if enable_hitl:
        builder.add_node("confirm", confirm_node)
        builder.add_edge("confirm", "tools")
        route_map["confirm"] = "confirm"
    builder.add_conditional_edges("agent", route_after_agent, route_map)

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

    # 入口链：START →(guard_input M7b)→(compress M6a)→(recall M6b)→ core_entry。
    # 按开关拼接，避免各组合写死。guard_input 排最前：进门第一件事就是扫用户输入。
    chain: list[str] = []
    if enable_guards:
        builder.add_node("guard_input", guard_input_node)
        chain.append("guard_input")
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
