"""M8 评测 harness 的**纯函数**离线测试：cassette 序列化、确定性指标、评分卡。

完全 hermetic——不联网、不碰 DB/Redis/Qdrant、不调 LLM，进 `make test`。
（端到端评测需本地 docker + cassette，由 `make eval` 手动跑，不放进单测。）
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from eval import metrics, report
from eval.judge import JudgeVerdict
from eval.replay import RecordingLLM, ReplayLLM
from eval.schema import Check, Expect, Scenario, ScenarioResult, Trace


# ---------- cassette 录制 → 回放 往返 ----------


class _FakeBound:
    def __init__(self, ret):
        self._ret = ret

    async def ainvoke(self, messages, *a, **k):  # noqa: ANN001
        return self._ret


class _FakeReal:
    """假的真实模型：bind_tools/with_structured_output 都返回固定结果，供录制往返测试。"""

    def __init__(self, ai=None, struct=None):
        self._ai, self._struct = ai, struct

    def bind_tools(self, tools, **k):  # noqa: ANN001
        return _FakeBound(self._ai)

    def with_structured_output(self, schema, **k):  # noqa: ANN001
        return _FakeBound(self._struct)


async def test_cassette_roundtrip_ai_message():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "book_trip", "args": {"item_id": "CA1831"}, "id": "c1"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    sink: list = []
    rec = RecordingLLM(_FakeReal(ai=ai), sink)
    await rec.bind_tools([]).ainvoke([])  # 录制
    assert sink[0]["kind"] == "ai"

    out = await ReplayLLM(sink).bind_tools([]).ainvoke([])  # 回放
    assert out.tool_calls[0]["name"] == "book_trip"
    assert out.usage_metadata["total_tokens"] == 15  # token 也被回放，成本仍可算


async def test_cassette_roundtrip_structured_output():
    verdict = JudgeVerdict(quality=4, faithful=True, reason="ok")
    sink: list = []
    rec = RecordingLLM(_FakeReal(struct=verdict), sink)
    await rec.with_structured_output(JudgeVerdict).ainvoke([])
    assert sink[0]["kind"] == "struct"

    rv = await ReplayLLM(sink).with_structured_output(JudgeVerdict).ainvoke([])
    assert rv.quality == 4 and rv.faithful is True


async def test_cassette_exhaustion_raises():
    import pytest

    rep = ReplayLLM([])
    with pytest.raises(RuntimeError):
        await rep.bind_tools([]).ainvoke([])  # 序列耗尽 → 明确报错提示重录


# ---------- 确定性指标 ----------


def _sc(category="tools", **expect_kw) -> Scenario:
    return Scenario(id="x", category=category, input="i", expect=Expect(**expect_kw))


def test_tool_recall_and_precision_pass():
    r = metrics.evaluate(
        _sc(tools_include=["search_flights"], tools_exclude=["book_trip"], max_steps=2),
        Trace(tools_called=["search_flights"], steps=1),
    )
    assert r.passed and r.tool_recall == 1.0


def test_forbidden_tool_fails_precision():
    r = metrics.evaluate(
        _sc(tools_include=["search_flights"], tools_exclude=["book_trip"]),
        Trace(tools_called=["search_flights", "book_trip"]),
    )
    assert not r.passed


def test_steps_over_budget_fails():
    r = metrics.evaluate(
        _sc(tools_include=["search_flights"], max_steps=1),
        Trace(tools_called=["search_flights"], steps=3),
    )
    assert not r.passed


def test_audit_and_not_executed():
    ok = metrics.evaluate(
        _sc("safety", audit_include=["injection.detected"], not_executed=["book_trip"]),
        Trace(audits=[{"action": "injection.detected", "detail": {}}]),
    )
    assert ok.passed
    bad = Trace(audits=[{"action": "tool.executed", "detail": {"tool": "book_trip"}}])
    bad.executed_high_risk = ["book_trip"]
    r = metrics.evaluate(_sc("safety", not_executed=["book_trip"]), bad)
    assert not r.passed  # 高危被真执行 → 失败


def test_pii_leak_in_final_fails():
    sc = _sc("safety", final_excludes=["13812345678"])
    assert metrics.evaluate(sc, Trace(final="已确认尾号 138****5678")).passed
    assert not metrics.evaluate(sc, Trace(final="您的号码 13812345678")).passed


def test_memory_recall_check():
    sc = _sc("memory", memory_recall_contains=["商务舱"])
    assert metrics.evaluate(sc, Trace(memory_context="相关历史记忆：偏好商务舱靠窗")).passed
    assert not metrics.evaluate(sc, Trace(memory_context="")).passed


def test_error_trace_fails():
    assert not metrics.evaluate(_sc(tools_include=["x"]), Trace(error="boom")).passed


# ---------- 评分卡 / 报告 ----------


def test_scorecard_and_markdown():
    results = [
        ScenarioResult("a", "tools", checks=[Check("tool_recall", True, True)], tool_recall=1.0),
        ScenarioResult("b", "safety", checks=[Check("audit:x", True, False)]),
    ]
    sc = report.build_scorecard(results, [{"id": "c", "reason": "no redis"}], {"mode": "replay"})
    assert sc["summary"]["total"] == 2 and sc["summary"]["passed"] == 1
    assert sc["summary"]["skipped"] == 1
    assert sc["by_category"]["tools"]["pass_rate"] == 1.0
    md = report.render_markdown(sc)
    assert "通过率" in md and "逐场景" in md


def test_scenario_parses_with_defaults():
    sc = Scenario.model_validate_json(
        '{"id":"a","category":"tools","input":"hi","expect":{"tools_include":["search_flights"]}}'
    )
    assert sc.principal == "alice" and sc.hitl is True
    assert sc.expect.tools_include == ["search_flights"]
