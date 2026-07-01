"""评分卡聚合 + 报告渲染：把每场景结果汇成 JSON 评分卡 + 人类可读 Markdown。

评分卡是"可迭代闭环"的凭据：改 prompt/检索参数/阈值后重跑，对比 pass_rate 与各维度分数，
用数据说"这次是真的更好了/悄悄弄坏了什么"，而不是凭手感。
"""

from __future__ import annotations

from eval.schema import ScenarioResult


def _rate(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def _avg(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def build_scorecard(results: list[ScenarioResult], skipped: list[dict], meta: dict) -> dict:
    passed = [r for r in results if r.passed]
    # 按类别
    cats: dict[str, list[ScenarioResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    by_category = {
        c: {
            "total": len(rs),
            "passed": sum(r.passed for r in rs),
            "pass_rate": _rate(sum(r.passed for r in rs), len(rs)),
        }
        for c, rs in sorted(cats.items())
    }
    # 聚合指标（各取有意义的子集）
    recalls = [r.tool_recall for r in results if r.tool_recall is not None]
    prec = [r.tool_precision_ok for r in results if r.tool_precision_ok is not None]
    mem_results = [r for r in results if r.category == "memory"]
    quals = [r.judge_quality for r in results if r.judge_quality is not None]
    faiths = [r.judge_faithful for r in results if r.judge_faithful is not None]
    metrics = {
        "tool_recall_avg": _avg(recalls),
        "tool_precision_pass_rate": _rate(sum(prec), len(prec)),
        "avg_steps": _avg([float(r.steps) for r in results if r.steps]),
        "total_tokens": sum(r.tokens for r in results),
        "memory_recall_pass_rate": _rate(sum(r.passed for r in mem_results), len(mem_results)),
        "judge_quality_avg": _avg([float(q) for q in quals]),
        "judge_faithful_rate": _rate(sum(bool(f) for f in faiths), len(faiths)),
    }
    return {
        "meta": meta,
        "summary": {
            "total": len(results),
            "passed": len(passed),
            "pass_rate": _rate(len(passed), len(results)),
            "skipped": len(skipped),
        },
        "by_category": by_category,
        "metrics": metrics,
        "skipped": skipped,
        "scenarios": [_scenario_row(r) for r in results],
    }


def _scenario_row(r: ScenarioResult) -> dict:
    failed = [f"{c.name}({c.detail})" for c in r.applicable_checks if not c.passed]
    return {
        "id": r.scenario_id,
        "category": r.category,
        "passed": r.passed,
        "steps": r.steps,
        "tokens": r.tokens,
        "tool_recall": r.tool_recall,
        "judge_quality": r.judge_quality,
        "judge_faithful": r.judge_faithful,
        "failed_checks": failed,
        "error": r.error,
    }


def render_markdown(sc: dict) -> str:
    m, s, met = sc["meta"], sc["summary"], sc["metrics"]
    lines = [
        "# 差旅 Agent 离线评测报告",
        "",
        f"- 模式：`{m.get('mode')}`　数据集：`{m.get('dataset')}`　生成：{m.get('generated_at', 'n/a')}",
        f"- **通过率：{s['passed']}/{s['total']} = {s['pass_rate']:.0%}**　（跳过 {s['skipped']}）",
        "",
        "## 分维度指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 工具召回率(平均) | {met['tool_recall_avg']:.0%} |",
        f"| 工具精确(未误触禁用)通过率 | {met['tool_precision_pass_rate']:.0%} |",
        f"| 平均执行步数 | {met['avg_steps']} |",
        f"| 记忆召回通过率 | {met['memory_recall_pass_rate']:.0%} |",
        f"| 判官质量(平均,1~5) | {met['judge_quality_avg']} |",
        f"| 判官 faithful 比例 | {met['judge_faithful_rate']:.0%} |",
        f"| 累计 token | {met['total_tokens']} |",
        "",
        "## 分类别通过率",
        "",
        "| 类别 | 通过/总数 | 通过率 |",
        "|---|---|---|",
    ]
    for c, v in sc["by_category"].items():
        lines.append(f"| {c} | {v['passed']}/{v['total']} | {v['pass_rate']:.0%} |")
    lines += [
        "",
        "## 逐场景",
        "",
        "| 场景 | 类别 | 结果 | 步数 | 质量 | 失败检查 |",
        "|---|---|---|---|---|---|",
    ]
    for row in sc["scenarios"]:
        mark = "✅" if row["passed"] else ("💥" if row["error"] else "❌")
        q = row["judge_quality"] if row["judge_quality"] is not None else "-"
        fails = "；".join(row["failed_checks"]) or ("运行异常" if row["error"] else "-")
        lines.append(
            f"| {row['id']} | {row['category']} | {mark} | {row['steps']} | {q} | {fails} |"
        )
    if sc["skipped"]:
        lines += ["", "## 跳过（依赖未就绪）", ""]
        for sk in sc["skipped"]:
            lines.append(f"- `{sk['id']}`：{sk['reason']}")
    lines.append("")
    return "\n".join(lines)
