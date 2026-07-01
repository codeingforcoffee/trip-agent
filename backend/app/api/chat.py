"""流式对话接口（M9a）：把 LangGraph 图的一轮运行翻译成 SSE 语义事件流。

三个设计要点（面试高频）：

1. **为什么用 SSE 不用 WebSocket**：聊天是【单向】的服务器→客户端推流，SSE 就是普通 HTTP
   长响应——天然穿透代理/负载均衡、EventSource 原生自动重连、比 WS 双向握手简单。SSE 的线格式
   极简：`event: <类型>\n` + `data: <JSON>\n` + 空行分隔（见 _sse_format）。
   注：浏览器原生 EventSource 只能 GET、不能带 Authorization 头，所以前端（M9c）用 fetch +
   ReadableStream 手动读流——那样才能 POST + 带 Bearer 头。此处后端对两种读法都兼容。

2. **stream_mode=["updates","messages"] 双模式**：一个 astream 同时拿两路——
   - "messages" 流 = LLM 的 token 增量 → 前端打字机（token 事件）；
   - "updates"  流 = 节点级状态更新 → 工具调用/返回、__interrupt__ 中断、引用（tool_call/…事件）。

3. **流式 PII 脱敏**：messages 流是 LLM 原文，若直接转发，PII 会在 guard_output 脱敏之前就
   过网。故每段 token 增量都过 StreamRedactor（回看缓冲，只在安全边界 flush，绝不把 PII
   从中间切开）。见 security/guards.py。

HITL（高危动作人工确认）与 SSE 的单向性：SSE 不能反向注入决策，所以确认走【二次请求】——
/chat 流到 interrupt 事件就结束，前端弹确认框，用户选择后调 /chat/resume；两次请求靠
thread_id（= tenant:user:conv）在 checkpointer 里接续同一会话状态。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.agent.graph import build_graph
from app.agent.tools import ALL_TOOLS
from app.core.deps import rate_limit, require_scopes
from app.core.logging import get_logger
from app.core.observability import record_usage
from app.core.security import SCOPE_CHAT, Identity
from app.security.guards import StreamRedactor

log = get_logger("app.api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])

# 返回带来源片段、值得作为“引用”展示的工具（RAG 政策检索）。
_CITATION_TOOLS = {"query_travel_policy"}

# SSE 响应头：关缓存 + 显式关代理缓冲（nginx 的 X-Accel-Buffering），否则中间层攒够一批才吐，
# 打字机效果全没了。Content-Type 由 media_type 设为 text/event-stream。
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class ChatRequest(BaseModel):
    message: str
    conversation_id: UUID | str | None = None  # 缺省用 "default" 会话线程


class ResumeRequest(BaseModel):
    conversation_id: UUID | str  # 必填：要接续哪个会话的中断
    approved: bool  # 用户对高危动作的批准/拒绝


def _truncate(s: str, n: int = 200) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _sse_format(event: str, data: dict) -> str:
    """把一个事件序列化成 SSE 线格式。ensure_ascii=False 让中文原样传，不被转义成 \\uXXXX。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_config(identity: Identity, conversation_id: UUID | str | None) -> dict:
    """把可信身份翻译成 LangGraph 的 RunnableConfig（身份放 configurable，模型碰不到）。

    thread_id = tenant:user:conv —— 短期记忆按它隔离，也保证“会话不串租户/用户”在命名层就成立。
    tenant_id/user_id 供 RAG 租户过滤与长期记忆；scopes 供 M7 工具授权。
    """
    conv = str(conversation_id) if conversation_id else "default"
    return {
        "configurable": {
            "thread_id": identity.thread_id(conv),
            "tenant_id": str(identity.tenant_id),
            "user_id": str(identity.user_id),
            "scopes": list(identity.scopes),
        }
    }


def _get_graph(request: Request):
    """取（并懒构建缓存）Agent 图。

    懒构建的用意：图依赖 LLM（get_llm 需 DEEPSEEK_API_KEY），而应用启动不该因缺 key 就崩——
    /health 等仍要能起（M0 的“降级不阻断”原则）。故图在首个 /chat 请求时才建、之后复用。
    checkpointer 用连接池版（lifespan 建于 app.state.checkpointer）。
    并发首请求可能重复建一次图（无害，最后一次写入生效），不值得为此加锁。
    """
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        from app.llm.deepseek import get_llm

        graph = build_graph(get_llm(), ALL_TOOLS, checkpointer=request.app.state.checkpointer)
        request.app.state.agent_graph = graph
    return graph


async def stream_chat_events(graph, config: dict, step_input) -> AsyncIterator[tuple[str, dict]]:
    """驱动图跑一轮，产出 (事件类型, 数据) 序列。step_input 为新消息或 Command(resume=…)。

    这是本模块的【可测核心】——纯逻辑、无 HTTP，测试用假图直接喂它、断言事件序列。
    事件类型：token / tool_call / tool_result / citation / interrupt / usage / done。
    """
    redactor = StreamRedactor()
    final_parts: list[str] = []
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0

    def _emit_token(raw: str) -> tuple[str, dict] | None:
        """把一段 LLM 原文增量过流式脱敏；有安全可吐的部分才产出 token 事件。"""
        safe = redactor.feed(raw)
        if safe:
            final_parts.append(safe)
            return "token", {"text": safe}
        return None

    async for mode, chunk in graph.astream(step_input, config, stream_mode=["updates", "messages"]):
        if mode == "messages":
            # chunk = (消息增量, 元数据)。只吐 agent 节点产生的最终答复文本；
            # 工具决策轮 content 为空、triage/摘要等其它节点的 token 都不吐给用户。
            msg_chunk, meta = chunk
            if meta.get("langgraph_node") == "agent" and isinstance(msg_chunk, AIMessageChunk):
                text = msg_chunk.content
                if text:
                    ev = _emit_token(str(text))
                    if ev:
                        yield ev
            continue

        # mode == "updates"：{节点名: 该节点的状态更新}
        for node, update in chunk.items():
            if node == "__interrupt__":
                # 命中 HITL 中断：先把缓冲里已成形的答复 flush 出去，再吐 interrupt 事件并结束本次流。
                tail = redactor.flush()
                if tail:
                    final_parts.append(tail)
                    yield "token", {"text": tail}
                payload = update[0].value if update else {}
                yield (
                    "interrupt",
                    dict(payload) if isinstance(payload, dict) else {"payload": payload},
                )
                # 中断也是本请求的一个结束点：把已累计的用量记进账本（供中间件核算成本）。
                # 只记账、不额外吐 usage 事件——保持"interrupt 即结束流"的 SSE 契约（前端只需弹确认框）。
                record_usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
                log.info("chat.interrupt", thread=config["configurable"]["thread_id"])
                return
            if not update:
                continue
            for m in update.get("messages", []):
                if isinstance(m, AIMessage) and node == "agent":
                    for call in m.tool_calls or []:
                        yield "tool_call", {"name": call["name"], "args": call["args"]}
                    if m.usage_metadata:
                        total_tokens += int(m.usage_metadata.get("total_tokens", 0) or 0)
                        input_tokens += int(m.usage_metadata.get("input_tokens", 0) or 0)
                        output_tokens += int(m.usage_metadata.get("output_tokens", 0) or 0)
                elif isinstance(m, ToolMessage):
                    yield "tool_result", {"name": m.name, "content": _truncate(str(m.content))}
                    if m.name in _CITATION_TOOLS:
                        yield (
                            "citation",
                            {"source": m.name, "snippet": _truncate(str(m.content), 400)},
                        )

    # 正常收尾：flush 脱敏缓冲余量 + 汇报 token 用量 + done（带脱敏后的完整答复，供非流式客户端）。
    tail = redactor.flush()
    if tail:
        final_parts.append(tail)
        yield "token", {"text": tail}
    # 记账本（供最外层 ObservabilityMiddleware 在流结束后核算 token 成本并写进 http.request 日志）。
    record_usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
    yield "usage", {"total_tokens": total_tokens}
    yield "done", {"final": "".join(final_parts)}


def _sse_response(event_gen: AsyncIterator[tuple[str, dict]]) -> StreamingResponse:
    """把事件序列包成 SSE 流响应；生成器内部异常兜成 error 事件，不让连接裸崩。"""

    async def framed() -> AsyncIterator[str]:
        try:
            async for ev_type, data in event_gen:
                yield _sse_format(ev_type, data)
        except Exception as e:  # noqa: BLE001 —— 流已开始，只能以事件形式告知前端出错
            log.exception("chat.stream_error", error=repr(e))
            yield _sse_format("error", {"message": "对话处理出错，请重试"})

    return StreamingResponse(framed(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
    identity: Identity = Depends(require_scopes(SCOPE_CHAT)),
    _rl=Depends(rate_limit()),
) -> StreamingResponse:
    """发起一轮对话，SSE 流式返回过程与答复。需 chat:write scope + 过限流。"""
    graph = _get_graph(request)
    config = _build_config(identity, body.conversation_id)
    step_input = {"messages": [HumanMessage(content=body.message)]}
    log.info("chat.start", thread=config["configurable"]["thread_id"])
    return _sse_response(stream_chat_events(graph, config, step_input))


@router.post("/resume")
async def chat_resume(
    body: ResumeRequest,
    request: Request,
    identity: Identity = Depends(require_scopes(SCOPE_CHAT)),
    _rl=Depends(rate_limit()),
) -> StreamingResponse:
    """接续被 HITL 中断的会话：带上用户对高危动作的决策，续跑同一 thread 并流式返回。"""
    graph = _get_graph(request)
    config = _build_config(identity, body.conversation_id)
    step_input = Command(resume={"approved": body.approved})
    log.info("chat.resume", thread=config["configurable"]["thread_id"], approved=body.approved)
    return _sse_response(stream_chat_events(graph, config, step_input))
