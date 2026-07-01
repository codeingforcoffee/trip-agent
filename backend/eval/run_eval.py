"""评测编排器：加载黄金场景 → 驱动图跑一遍 → 采集轨迹 → 确定性打分 + 判官评质 → 出评分卡。

三种模式（对应 Makefile 三个目标）：
  - replay（默认，make eval）：agent 与 judge 的 LLM 全部走 cassette 回放 → 完全离线、确定、免费。
    注意：**图里的工具仍真跑**（book_trip 走 Redis、RAG 查 Qdrant）——cassette 只冻结 LLM 这一个
    "非确定 + 联网 + 花钱"的依赖，本地 docker 依赖属于 M0 已锁定的可复现环境。
  - live（make eval-live）：真跑 DeepSeek，评"当前 prompt 下的真实质量"，不写 cassette。
  - record（make eval-record）：真跑并把每次 LLM 调用按序录进 cassette，供以后回放。

依赖未就绪的场景（booking 需 Redis、rag/memory 需 Qdrant）会被**显式跳过并记录**（不静默略过，
也不算失败）——报告里能看到跳了哪些、为什么。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import app.agent.graph as graph_mod
from app.agent.graph import build_graph
from app.agent.tools import ALL_TOOLS
from app.core.logging import get_logger, setup_logging
from eval import judge as judge_mod
from eval import metrics, report
from eval.replay import RecordingLLM, ReplayLLM
from eval.schema import Scenario, Trace

log = get_logger("eval.run_eval")

EVAL_DIR = Path(__file__).resolve().parent
DATASETS, CASSETTES, REPORTS = EVAL_DIR / "datasets", EVAL_DIR / "cassettes", EVAL_DIR / "reports"

# 各类别真正需要的外部依赖（不在表内=只用 mock 工具/护栏，无需 docker 依赖）
CATEGORY_NEEDS = {"booking": {"redis"}, "rag": {"qdrant"}, "memory": {"qdrant"}}


class AuditCollector:
    """替换 graph 的 record_audit：把安全事件收进内存，供确定性断言（也顺带避开 Postgres 依赖）。"""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def reset(self) -> None:
        self.events = []

    async def __call__(self, tenant_id, user_id, action, detail) -> None:  # noqa: ANN001
        self.events.append({"action": action, "detail": detail or {}})


def load_scenarios(dataset: str) -> list[Scenario]:
    path = DATASETS / f"{dataset}.jsonl"
    scenarios = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            scenarios.append(Scenario.model_validate_json(line))
    return scenarios


async def resolve_principals() -> dict[str, tuple]:
    """把 alice/bob 解析成 (tenant_id, user_id, scopes)——模拟"登录后从 JWT 拿到的可信身份"。"""
    from app.agent.cli import _resolve_ids

    alice = await _resolve_ids("acme", "alice@acme.com")
    bob = await _resolve_ids("globex", "bob@globex.com")
    if not alice[0] or not bob[0]:
        raise RuntimeError("解析不到 alice/bob 身份——请先 make up && make seed")
    return {"alice": alice, "bob": bob}


async def infra_status() -> set[str]:
    """探活 redis/qdrant，返回**可用**依赖集合。用于跳过依赖未就绪的场景。"""
    available: set[str] = set()
    try:
        from app.infra.redis_client import get_redis_client

        await get_redis_client().ping()
        available.add("redis")
    except Exception as e:  # noqa: BLE001
        log.info("infra.redis_down", error=repr(e))
    try:
        from app.infra.qdrant import get_qdrant_client

        await get_qdrant_client().get_collections()
        available.add("qdrant")
    except Exception as e:  # noqa: BLE001
        log.info("infra.qdrant_down", error=repr(e))
    return available


async def _seed_memory(scenario: Scenario, principal: tuple) -> None:
    from app.agent import memory as mem

    tenant_id, user_id, _ = principal
    # dedup 阈值给高：同一条重复 seed 会被判重跳过，多次跑 eval 不会累积垃圾记忆
    await mem.remember_semantic(scenario.seed_memory, tenant_id, user_id, dedup_threshold=0.99)


async def run_scenario(
    scenario: Scenario, principal: tuple, llm, collector: AuditCollector
) -> Trace:
    """驱动一个场景跑完整一轮（含 HITL resume），把可观测信号采进 Trace。"""
    tenant_id, user_id, scopes = principal
    if scenario.seed_memory:
        await _seed_memory(scenario, principal)

    graph = build_graph(
        llm,
        ALL_TOOLS,
        checkpointer=MemorySaver(),  # 用内存 checkpointer：eval 不依赖 Postgres，也隔离各场景
        enable_triage=scenario.triage,
        enable_compress=False,
        enable_memory=scenario.memory,
        enable_hitl=scenario.hitl,
        enable_guards=True,
    )
    config = {
        "configurable": {
            "thread_id": f"eval-{scenario.id}",
            "tenant_id": tenant_id,
            "user_id": user_id,
            "scopes": scopes,
        }
    }
    collector.reset()
    trace = Trace()
    step_input: object = {"messages": [HumanMessage(content=scenario.input)]}
    try:
        while step_input is not None:
            resume = None
            async for chunk in graph.astream(step_input, config, stream_mode="updates"):
                for node, upd in chunk.items():
                    if node == "__interrupt__":
                        trace.interrupted = True
                        resume = Command(resume={"approved": scenario.hitl_approve})
                    elif node == "recall":
                        trace.memory_context = (upd or {}).get(
                            "memory_context", ""
                        ) or trace.memory_context
                    else:
                        for m in (upd or {}).get("messages", []):
                            if isinstance(m, AIMessage):
                                if node == "agent":
                                    trace.steps += 1  # agent 每被调用一次=一步 ReAct 推理
                                    trace.tokens += int(
                                        (m.usage_metadata or {}).get("total_tokens", 0)
                                    )
                                    for c in m.tool_calls or []:
                                        trace.tools_called.append(c["name"])
                                if m.content:
                                    trace.final = str(
                                        m.content
                                    )  # 末次非空 AI 文本；guard_output 会覆盖成脱敏版
                            elif isinstance(m, ToolMessage):
                                trace.tool_outputs.append(str(m.content))
            step_input = resume
    except Exception as e:  # noqa: BLE001 —— 单场景异常判 fail，不炸整场评测
        trace.error = repr(e)
    trace.audits = list(collector.events)
    trace.executed_high_risk = [
        a["detail"].get("tool") for a in trace.audits if a["action"] == "tool.executed"
    ]
    return trace


async def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="差旅 Agent 离线评测（M8）")
    parser.add_argument("--mode", choices=["replay", "live", "record"], default="replay")
    parser.add_argument("--dataset", default="trip_eval")
    args = parser.parse_args()

    scenarios = load_scenarios(args.dataset)
    principals = await resolve_principals()
    available = await infra_status()
    log.info("eval.start", mode=args.mode, scenarios=len(scenarios), infra=sorted(available))

    collector = AuditCollector()
    graph_mod.record_audit = collector  # 全局替换：所有 _audit 落进内存收集器

    cassette_path = CASSETTES / f"{args.dataset}.json"
    if args.mode == "replay":
        if not cassette_path.exists():
            raise SystemExit(f"缺 cassette：{cassette_path}，请先 make eval-record")
        cassette = json.loads(cassette_path.read_text(encoding="utf-8"))
    else:
        from app.llm.deepseek import get_llm

        real_llm = get_llm()
        cassette = {}

    results, skipped = [], []
    new_cassette: dict = {}
    for sc in scenarios:
        needs = CATEGORY_NEEDS.get(sc.category, set())
        missing = needs - available
        if missing:
            skipped.append({"id": sc.id, "reason": f"依赖未就绪：{sorted(missing)}"})
            log.info("eval.skip", id=sc.id, missing=sorted(missing))
            continue

        # 按模式给出 agent / judge 的 LLM（回放各取一条录制序列；录制/live 真跑）
        if args.mode == "replay":
            entry = cassette.get(sc.id)
            if entry is None:
                skipped.append({"id": sc.id, "reason": "cassette 无此场景，请重录"})
                continue
            agent_llm, judge_llm = ReplayLLM(entry["agent"]), ReplayLLM(entry["judge"])
            agent_sink = judge_sink = None
        else:
            agent_sink, judge_sink = [], []
            agent_llm = RecordingLLM(real_llm, agent_sink)
            judge_llm = RecordingLLM(real_llm, judge_sink)

        trace = await run_scenario(sc, principals[sc.principal], agent_llm, collector)
        verdict = await judge_mod.judge(sc, trace, judge_llm)
        result = metrics.evaluate(sc, trace)
        if verdict is not None:
            result.judge_quality = verdict.quality
            result.judge_faithful = verdict.faithful
            result.judge_reason = verdict.reason
        results.append(result)
        if args.mode != "replay":
            new_cassette[sc.id] = {"agent": agent_sink, "judge": judge_sink}
        log.info("eval.scenario", id=sc.id, passed=result.passed, steps=result.steps)

    if args.mode == "record":
        CASSETTES.mkdir(exist_ok=True)
        cassette_path.write_text(
            json.dumps(new_cassette, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("eval.cassette_written", path=str(cassette_path), scenarios=len(new_cassette))

    meta = {
        "mode": args.mode,
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    scorecard = report.build_scorecard(results, skipped, meta)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{args.dataset}.json").write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = report.render_markdown(scorecard)
    (REPORTS / f"{args.dataset}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    asyncio.run(main())
