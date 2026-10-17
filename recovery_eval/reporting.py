"""Write reproducible result bundles without external plotting dependencies."""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import statistics
from typing import Any


LABELS = {
    "bounded_retry": "有界重试", "verify_then_retry": "查询后重试",
    "idempotent_retry": "稳定幂等键重试", "clean": "无故障对照",
    "before_commit_once": "提交前失败", "ack_lost": "提交后丢失响应",
    "late_commit": "延迟提交", "stale_read": "丢失响应＋陈旧查询",
    "duplicate_delivery": "重复投递",
}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count episodes, keeping repeated parameterizations distinct from tasks."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results:
        grouped[(row["policy"], "all")].append(row)
        grouped[(row["policy"], row["fault"])].append(row)
    metrics = []
    for (policy, fault), rows in sorted(grouped.items()):
        n = len(rows)
        metrics.append({
            "policy": policy, "fault": fault, "episodes": n,
            "successes": sum(bool(r["success"]) for r in rows),
            "success_rate": sum(bool(r["success"]) for r in rows) / n,
            "duplicate_episode_rate": sum(r["duplicates"] > 0 for r in rows) / n,
            "missing_episode_rate": sum(r["missing"] > 0 for r in rows) / n,
            "false_success_rate": sum(bool(r["false_success"]) for r in rows) / n,
            "mean_tool_calls": statistics.mean(r["tool_calls"] for r in rows),
            "mean_logical_ticks": statistics.mean(r["logical_ticks"] for r in rows),
        })
    return {"episodes": len(results),
            "task_payloads": len({r["task_id"] for r in results}),
            "fault_conditions": len({r["fault"] for r in results}),
            "seeds": len({r["seed"] for r in results}), "metrics": metrics}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def diagnosis_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build observable diagnosis cases, not partially observed recovery tasks.

    The final settled-state tool observation is deliberately given to a
    diagnosis model. Hidden oracle labels are stored separately for scoring.
    """
    rows = []
    for result in results:
        records = result["final_state"]["records"]
        notifications = result["final_state"]["notifications"]
        trace = [{"tool": "inspect_settled_state", "response": {
            "records": records, "notifications": notifications}}]
        if not isinstance(result.get("task_payload"), dict):
            raise ValueError("diagnosis export requires the original task_payload; never infer it from IDs")
        request = {"goal": "Create exactly one correct service record and exactly one notification linked to it.",
                   **result["task_payload"]}
        # Commit, fault, policy, and oracle annotations do not enter the prompt.
        rows.append({"id": f"{result['case_id']}:{result['policy']}",
                     "request": request, "trace": trace,
                     "expected": {"duplicates": result["duplicates"],
                                  "missing": result["missing"],
                                  "success": result["success"]}})
    return rows


def diagnosis_controls(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute known-answer negative controls through the same public tools."""
    from .core import RecoveryEnvironment

    results = []
    selected = [c for c in cases if c["fault"] == "clean" and c["seed"] == 0]
    for case in selected:
        for variant in ("empty", "record_only", "wrong_message", "wrong_link"):
            env = RecoveryEnvironment(case)
            p = env.task
            record = None
            if variant != "empty":
                response = env.invoke("create_record", {k: p[k] for k in ("task_id", "title", "body")})
                record = response["result"]["record_id"]
            if variant in ("wrong_message", "wrong_link"):
                env.invoke("send_notification", {
                    "task_id": p["task_id"],
                    "record_id": "nonexistent-record" if variant == "wrong_link" else record,
                    "recipient": p["recipient"],
                    "message": "Incorrect message" if variant == "wrong_message" else p["message"],
                })
            result = env.finish(False, policy="known_answer_control", evaluation_mode="known_answer_control")
            result["case_id"] += ":" + variant
            results.append(result)
    return results


def write_artifacts(results: list[dict[str, Any]], out: str | Path,
                    metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Save raw episodes, summaries, example traces and a self-contained report."""
    if not results:
        raise ValueError("results must not be empty")
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    manifest = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "backend": "local", "cloud_execution": False,
                "uses_language_model": any(r.get("uses_language_model", False) for r in results),
                "experiment_kind": ("live_model_recovery" if any(r.get("uses_language_model", False) for r in results)
                                    else "scripted_policy_simulation"), **(metadata or {})}
    is_scripted = not manifest["uses_language_model"]
    title = "首轮规则基线实验" if is_scripted else "真实模型恢复评测"
    experiment_note = ("这是<strong>手写规则策略的确定性仿真实验</strong>，不是 LLM 成绩。" if is_scripted
                       else "这是<strong>真实模型工具调用实验</strong>；模型、提示词和调用记录保存在原始结果中。模型格式错误、预算耗尽及端点错误需分别检查 run_status，不能统一解释为能力不足。")
    reproduction = ("python -m recovery_eval demo --out results/local" if is_scripted
                    else "python -m recovery_eval.api_agent --help")
    intro = ("同一目标、同一工具契约、同一故障种子，对照三种恢复策略。" if is_scripted
             else "固定业务目标、工具契约与故障条件，记录真实模型的工具调用及最终业务结果。")
    with (directory / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _json_dump(directory / "summary.json", summary)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["metrics"][0]))
        writer.writeheader()
        writer.writerows(summary["metrics"])
    examples = []
    for policy in sorted({r["policy"] for r in results}):
        candidates = [r for r in results if r["policy"] == policy]
        examples.extend([r for r in candidates if not r["success"]][:2])
        examples.extend([r for r in candidates if r["success"]][:1])
    _json_dump(directory / "example_traces.json", examples)
    rows_html = "".join(
        "<tr><td>" + html.escape(LABELS.get(m["policy"], m["policy"])) + "</td><td>" +
        html.escape("全部条件" if m["fault"] == "all" else LABELS.get(m["fault"], m["fault"])) +
        f"</td><td>{m['episodes']}</td><td>{m['success_rate']:.1%}</td>" +
        f"<td>{m['duplicate_episode_rate']:.1%}</td><td>{m['missing_episode_rate']:.1%}</td>" +
        f"<td>{m['false_success_rate']:.1%}</td><td>{m['mean_tool_calls']:.2f}</td></tr>"
        for m in summary["metrics"])
    details = "".join(
        "<details><summary>" + html.escape(r["case_id"] + " · " + r["policy"]) +
        f" · {'通过' if r['success'] else '失败'}</summary><pre>" +
        html.escape(json.dumps(r, ensure_ascii=False, indent=2)) + "</pre></details>"
        for r in examples)
    report = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Recovery Eval Lite · 实验报告</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;margin:0;background:#f3f6fa;color:#162438}}main{{max-width:1160px;margin:auto;padding:40px 24px}}h1{{font-size:34px;line-height:1.25}}h2{{margin-top:32px}}.tag{{font-size:12px;letter-spacing:2px;color:#286a67}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card,.note,details{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:18px}}.card strong{{display:block;font-size:30px}}.note{{border-left:5px solid #21897f}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #e1e7ee}}th{{background:#e5edf5}}.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f4f7fa;padding:12px;max-height:500px;overflow:auto}}details{{margin:12px 0}}a{{color:#126e76}}footer{{font-size:13px;color:#52647a;margin-top:30px}}@media(max-width:650px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:24px 14px}}}}</style>
<main><div class="tag">REPRODUCIBLE EVALUATION / VERSION 0.1.0</div>
<h1>Agent 故障恢复<br>{title}</h1>
<p>{intro}以最终业务状态和已提交操作账本评分。</p>
<div class="cards"><div class="card"><strong>{summary['episodes']}</strong>实际执行回合</div><div class="card"><strong>{summary['task_payloads']}</strong>业务参数样例</div><div class="card"><strong>{summary['fault_conditions']}</strong>条件（含无故障对照）</div><div class="card"><strong>{summary['seeds']}</strong>调度种子</div></div>
<h2>1. 本次运行状态</h2><div class="note">执行后端：{html.escape(str(manifest['backend']))}。{experiment_note}当前两步工作流为“创建记录 → 发送关联通知”；不同业务参数不等于不同任务类型。虚拟时钟不等于真实网络时延。云端计算任务：{'已执行' if manifest.get('cloud_execution') else '未执行'}。</div>
<h2>2. 实测结果</h2><div class="scroll"><table><thead><tr><th>策略</th><th>条件</th><th>回合</th><th>完成率</th><th>重复操作回合率</th><th>漏操作回合率</th><th>错误宣称成功率</th><th>平均工具调用</th></tr></thead><tbody>{rows_html}</tbody></table></div>
<h2>3. 如何解释结果</h2><p>完成必须同时满足：恰好一条正确记录、恰好一条正确通知、通知关联正确记录。延迟提交会在评分前全部送达。幂等策略使用模拟服务本来就支持的稳定操作键；优势属于策略与服务端契约的共同作用，不代表模型智能提升或现实系统保证。种子只改变故障目标与提交延迟；本报告不给出跨业务或跨模型泛化结论。</p>
<h2>4. 可核查的轨迹</h2><p>展开可查看工具调用、超时、延迟提交、操作账本与最终判定。失败例优先展示，每种策略另给一个成功例。</p>{details}
<h2>5. 复现与原始数据</h2><p><code>{reproduction}</code></p>
<p><a href="episodes.jsonl">逐回合 JSONL</a> · <a href="summary.csv">指标 CSV</a> · <a href="summary.json">汇总 JSON</a> · <a href="manifest.json">运行记录</a></p>
<footer>生成于 {html.escape(manifest['created_utc'])}。模型轨迹诊断是独立实验，见项目中的 results/local_model；是否完成以其实际结果文件为准。</footer></main></html>"""
    (directory / "report.html").write_text(report, encoding="utf-8")
    manifest["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(directory.iterdir()) if p.is_file() and p.name != "manifest.json"}
    _json_dump(directory / "manifest.json", manifest)
    return summary
