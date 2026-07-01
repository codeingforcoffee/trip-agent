"""M9a 流式对话（SSE）单测：全部 hermetic —— 不连 DB/Redis/网络，用假图喂脚本化事件。

覆盖三块可测核心：
  1. StreamRedactor：流式 PII 脱敏（跨 token 切分不漏、中文正常流、尾部 flush）；
  2. _sse_format：SSE 线格式（含中文不转义）；
  3. stream_chat_events：把图运行翻译成 token/tool_call/tool_result/citation/interrupt/usage/done。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.api.chat import _sse_format, stream_chat_events
from app.security.guards import StreamRedactor

# ————————————————————————— 测试替身 —————————————————————————


class _Interrupt:
    """模拟 LangGraph __interrupt__ chunk 里的 Interrupt 元素（只需 .value）。"""

    def __init__(self, value: dict) -> None:
        self.value = value


class FakeGraph:
    """假图：astream 按脚本吐 (mode, chunk)。签名对齐真图 graph.astream(input, config, stream_mode=)。"""

    def __init__(self, script: list[tuple[str, object]]) -> None:
        self._script = script

    def astream(self, step_input, config, stream_mode=None):  # noqa: ANN001
        script = self._script

        async def gen():
            for item in script:
                yield item

        return gen()


async def _collect(graph, step_input=None) -> list[tuple[str, dict]]:
    return [
        ev
        async for ev in stream_chat_events(graph, {"configurable": {"thread_id": "t"}}, step_input)
    ]


# ————————————————————————— StreamRedactor —————————————————————————


def _run_redactor(chunks: list[str]) -> str:
    red = StreamRedactor()
    out = "".join(red.feed(c) for c in chunks)
    return out + red.flush()


def test_redactor_masks_pii_split_across_chunks():
    # 手机号被切成 "138" | "12345678" 两段——逐 token 脱敏会漏，回看缓冲不漏。
    out = _run_redactor(["我的手机号是138", "12345678", "请核对"])
    assert out == "我的手机号是138****5678请核对"


def test_redactor_flushes_trailing_pii_without_boundary():
    # 末尾就是 PII、后面没有边界字符——靠 flush 兜底脱敏。
    out = _run_redactor(["联系", "13812345678"])
    assert out == "联系138****5678"


def test_redactor_masks_email():
    out = _run_redactor(["邮箱", "zhang@acme.com", "完成"])
    assert out == "邮箱z***@acme.com完成"


def test_redactor_passes_clean_chinese_through():
    # 纯中文无 PII：原样流出（且不整段憋着——每段都能 flush，保证打字机效果）。
    red = StreamRedactor()
    assert red.feed("上海住宿") == "上海住宿"  # 中文即安全边界，立即吐出
    assert red.feed("报销上限600元") == "报销上限600元"  # 600 非 PII 模式，不误伤
    assert red.flush() == ""


# ————————————————————————— _sse_format —————————————————————————


def test_sse_format_wire_and_unicode():
    line = _sse_format("token", {"text": "你好"})
    # 线格式：event: 类型 / data: JSON / 空行分隔；中文不被转义成 \uXXXX
    assert line == 'event: token\ndata: {"text": "你好"}\n\n'


# ————————————————————————— stream_chat_events —————————————————————————


async def test_stream_events_rag_answer_sequence():
    """一轮 RAG 问答：工具调用 → 工具返回 → 引用 → 流式 token → 用量 → done。"""
    script = [
        (
            "updates",
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "query_travel_policy",
                                    "args": {"city": "上海"},
                                    "id": "c1",
                                }
                            ],
                        )
                    ]
                }
            },
        ),
        (
            "updates",
            {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="上海住宿报销上限600元/晚。[来源:差旅政策v2]",
                            name="query_travel_policy",
                            tool_call_id="c1",
                        )
                    ]
                }
            },
        ),
        ("messages", (AIMessageChunk(content="上海住宿"), {"langgraph_node": "agent"})),
        ("messages", (AIMessageChunk(content="报销上限600元/晚。"), {"langgraph_node": "agent"})),
        (
            "updates",
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="上海住宿报销上限600元/晚。",
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 8,
                                "total_tokens": 18,
                            },
                        )
                    ]
                }
            },
        ),
    ]
    events = await _collect(FakeGraph(script), {"messages": []})
    types = [t for t, _ in events]
    assert types == [
        "tool_call",
        "tool_result",
        "citation",
        "token",
        "token",
        "usage",
        "done",
    ]
    by = {t: d for t, d in events}
    assert by["tool_call"]["name"] == "query_travel_policy"
    assert by["citation"]["source"] == "query_travel_policy"
    assert by["usage"]["total_tokens"] == 18
    assert by["done"]["final"] == "上海住宿报销上限600元/晚。"


async def test_stream_events_masks_pii_in_token_stream():
    """token 流里的 PII 必须在过网前脱敏（跨 chunk 切分也不漏）。"""
    script = [
        ("messages", (AIMessageChunk(content="您的手机号是138"), {"langgraph_node": "agent"})),
        ("messages", (AIMessageChunk(content="12345678"), {"langgraph_node": "agent"})),
        ("messages", (AIMessageChunk(content="，已确认。"), {"langgraph_node": "agent"})),
    ]
    events = await _collect(FakeGraph(script), {"messages": []})
    text = "".join(d["text"] for t, d in events if t == "token")
    assert "13812345678" not in text
    assert "138****5678" in text
    done = [d for t, d in events if t == "done"][0]
    assert "13812345678" not in done["final"]


async def test_stream_events_non_agent_tokens_ignored():
    """只有 agent 节点的 token 才吐给用户；triage/摘要等其它节点的不吐。"""
    script = [
        ("messages", (AIMessageChunk(content="内部分诊思考"), {"langgraph_node": "triage"})),
        ("messages", (AIMessageChunk(content="正式答复"), {"langgraph_node": "agent"})),
    ]
    events = await _collect(FakeGraph(script), {"messages": []})
    tokens = [d["text"] for t, d in events if t == "token"]
    assert tokens == ["正式答复"]


async def test_stream_events_interrupt_ends_stream():
    """命中 HITL 中断：吐 tool_call + interrupt 后即结束，不再有 usage/done。"""
    script = [
        (
            "updates",
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "book_trip", "args": {"flight": "CA1831"}, "id": "b1"}
                            ],
                        )
                    ]
                }
            },
        ),
        (
            "updates",
            {
                "__interrupt__": (
                    _Interrupt(
                        {
                            "message": "确认下单？",
                            "actions": [{"tool": "book_trip", "args": {"flight": "CA1831"}}],
                        }
                    ),
                )
            },
        ),
        # 中断后即便脚本还有内容，也不应被消费（stream_chat_events 应已 return）
        ("messages", (AIMessageChunk(content="不该出现"), {"langgraph_node": "agent"})),
    ]
    events = await _collect(FakeGraph(script), {"messages": []})
    types = [t for t, _ in events]
    assert types == ["tool_call", "interrupt"]
    interrupt = [d for t, d in events if t == "interrupt"][0]
    assert interrupt["message"] == "确认下单？"
