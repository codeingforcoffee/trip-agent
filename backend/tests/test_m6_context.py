"""M6a 测试：上下文压缩（全部离线，不联网）。

分两层：
  1. context.py 纯逻辑——估算 / 安全切点 / maybe_compress（用假摘要器驱动）；
  2. 图级别——用假 LLM 跑通"超预算 → compress 节点摘要并移除旧消息 → agent 继续"。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.context import estimate_tokens, maybe_compress
from app.agent.graph import build_graph
from app.agent.tools import ALL_TOOLS
from app.core.config import settings


async def _fake_summarize(evicted, old_summary):  # noqa: ANN001
    """确定性假摘要器：记录被并入的条数与旧摘要，便于断言。"""
    return f"[摘要|并入{len(evicted)}条|旧:{old_summary or '空'}]"


def _conversation_with_ids() -> list:
    """造三轮对话，含一组 tool_call+ToolMessage，每条带显式 id（RemoveMessage 需要 id）。"""
    return [
        HumanMessage(content="我想去上海出差", id="h1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_trains", "args": {}, "id": "tc1"}],
            id="a1",
        ),
        ToolMessage(content="车次结果若干", tool_call_id="tc1", id="t1"),
        AIMessage(content="为你找到几趟高铁", id="a2"),
        HumanMessage(content="从北京出发", id="h2"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_trains", "args": {}, "id": "tc2"}],
            id="a3",
        ),
        ToolMessage(content="北京到上海车次", tool_call_id="tc2", id="t3"),
        AIMessage(content="推荐 G1 次", id="a4"),
        HumanMessage(content="下周一的", id="h3"),
        AIMessage(content="好的", id="a5"),
    ]


# ---------- 纯逻辑 ----------


def test_estimate_tokens_grows_with_content():
    short = [HumanMessage(content="你好")]
    long = [HumanMessage(content="差旅报销政策" * 50)]
    assert estimate_tokens(long) > estimate_tokens(short) > 0


async def test_no_compress_under_budget():
    msgs = _conversation_with_ids()
    out = await maybe_compress(msgs, "", budget=10_000, keep_last=6, summarize=_fake_summarize)
    assert out is None  # 没超预算 → 不动


async def test_compress_evicts_old_keeps_recent_at_human_boundary():
    msgs = _conversation_with_ids()  # 10 条
    out = await maybe_compress(msgs, "", budget=1, keep_last=2, summarize=_fake_summarize)
    assert out is not None
    # keep_last=2 的目标位是 index 8（h3），正好是 HumanMessage → 淘汰前 8 条（前两轮完整轮次）
    assert out.evicted_count == 8
    assert out.removed_ids == ["h1", "a1", "t1", "a2", "h2", "a3", "t3", "a4"]
    assert out.summary == "[摘要|并入8条|旧:空]"


async def test_safe_cut_never_orphans_tool_message():
    """即便 keep_last 落在某轮中间（如 a4），也会向前snap到 HumanMessage，保完整轮次。"""
    msgs = _conversation_with_ids()
    # keep_last=3 → 目标 index 7（a4，AI），不能从这切（会孤立 tc2 的 ToolMessage）
    out = await maybe_compress(msgs, "", budget=1, keep_last=3, summarize=_fake_summarize)
    assert out is not None
    # 向前snap到 index 4（h2）→ 只淘汰第一轮 4 条，保留后两轮完整
    assert out.evicted_count == 4
    assert out.removed_ids == ["h1", "a1", "t1", "a2"]


async def test_merges_into_existing_summary():
    msgs = _conversation_with_ids()
    out = await maybe_compress(msgs, "旧摘要内容", budget=1, keep_last=2, summarize=_fake_summarize)
    assert out is not None
    assert "旧:旧摘要内容" in out.summary  # 旧摘要被传入摘要器融合


# ---------- 图级别（假 LLM，不联网）----------


class _FakeAgent:
    async def ainvoke(self, messages):  # noqa: ANN001
        return AIMessage(content="收到，继续为你处理。")  # 无 tool_calls → 直接 END


class _FakeLLM:
    """bind_tools 给 agent 用；ainvoke 给压缩摘要器用。triage 关闭，无需结构化输出。"""

    def bind_tools(self, tools):  # noqa: ANN001
        return _FakeAgent()

    async def ainvoke(self, messages):  # noqa: ANN001
        return AIMessage(content="[压缩摘要]")


async def test_graph_compress_node_fires_and_shrinks_history(monkeypatch):
    # 把预算压到极低、保留最近 4 条，确保多消息输入必触发压缩
    monkeypatch.setattr(settings, "context_token_budget", 50)
    monkeypatch.setattr(settings, "compress_keep_last", 4)

    graph = build_graph(_FakeLLM(), ALL_TOOLS, enable_triage=False, enable_compress=True)
    # 8 条历史（4 轮 H/A），足以超 50 token 预算
    history = []
    for i in range(4):
        history.append(HumanMessage(content=f"第{i}轮问题：差旅报销标准是怎样的？"))
        history.append(AIMessage(content=f"第{i}轮回答：这是相关说明内容。"))

    out = await graph.ainvoke({"messages": history})

    assert out.get("summary"), "应生成滚动摘要"
    # 旧消息被移除：输入 8 条历史，压缩后保留 < 8，再 + 1 条 agent 回复
    non_system = [m for m in out["messages"]]
    assert len(non_system) < 9
    assert out["messages"][-1].content == "收到，继续为你处理。"
