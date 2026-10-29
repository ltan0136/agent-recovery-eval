"""Portable v2 run artifacts and a self-contained review report."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import platform
import statistics
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.py"))}


def seed_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """pass^k across seeds: a (workflow, variant, condition) family passes only if every seed succeeds.

    Seeds move the fault onto a different step and change stale/late schedules, so a policy
    that only recovers on some steps scores lower here than on the per-episode success rate.
    ``k`` is the smallest number of seeds observed in any family of the slice.
    """
    families = defaultdict(list)
    for row in rows:
        families[(row["agent_id"], row["workflow"], row["variant_id"], row["fault"])].append(bool(row["success"]))
    slices = defaultdict(list)
    for (agent, _workflow, _variant, fault), outcomes in families.items():
        slices[(agent, "all")].append(outcomes)
        slices[(agent, fault)].append(outcomes)
    result = []
    for (agent, fault), groups in sorted(slices.items()):
        passed = sum(all(outcomes) for outcomes in groups)
        episodes = [ok for outcomes in groups for ok in outcomes]
        result.append({"agent_id": agent, "fault": fault, "families": len(groups),
                       "k": min(len(outcomes) for outcomes in groups),
                       "all_seeds_successes": passed, "all_seeds_success_rate": passed / len(groups),
                       "per_episode_success_rate": sum(episodes) / len(episodes)})
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty run")
    groups = defaultdict(list)
    agent_kinds = {}
    seen = set()
    for row in rows:
        identity = (row["agent_id"], row["case_id"])
        if identity in seen:
            raise ValueError(f"Duplicate case within agent: {identity}")
        seen.add(identity)
        previous_kind = agent_kinds.setdefault(row["agent_id"], row["agent_kind"])
        if previous_kind != row["agent_kind"]:
            raise ValueError(f"An agent_id cannot combine different agent kinds: {row['agent_id']}")
        if type(row.get("success")) is not bool or type(row.get("false_success")) is not bool:
            raise ValueError("success and false_success must be JSON booleans")
        groups[(row["agent_id"], "all", "all")].append(row)
        groups[(row["agent_id"], row["workflow"], "all")].append(row)
        groups[(row["agent_id"], "all", row["fault"])].append(row)
        groups[(row["agent_id"], row["workflow"], row["fault"])].append(row)
    metrics = []
    for (agent, workflow, fault), items in sorted(groups.items()):
        n = len(items)
        calls = [call for item in items for call in item.get("model_calls", [])]
        metrics.append({"agent_id": agent, "agent_kind": agent_kinds[agent],
                        "workflow": workflow, "fault": fault, "episodes": n,
                        "successes": sum(bool(r["success"]) for r in items),
                        "success_rate": sum(bool(r["success"]) for r in items) / n,
                        "duplicate_episode_rate": sum(r["duplicates"] > 0 for r in items) / n,
                        "missing_episode_rate": sum(r["missing"] > 0 for r in items) / n,
                        "false_success_rate": sum(bool(r["false_success"]) for r in items) / n,
                        "mean_tool_calls": statistics.mean(r["tool_calls"] for r in items),
                        "model_calls": len(calls),
                        "completion_tokens": sum((c.get("usage") or {}).get("completion_tokens", 0) or 0 for c in calls),
                        "invalid_model_responses": sum(c.get("status") == "invalid_action" for c in calls),
                        "truncated_model_responses": sum(bool((c.get("metadata") or {}).get("generation_truncated")) for c in calls),
                        "infrastructure_error_episodes": sum(r.get("run_status") == "infrastructure_error" for r in items),
                        "provider_error_episodes": sum(r.get("run_status") in ("provider_error", "model_error", "infrastructure_error") for r in items)})
    return {"schema_version": "2.0", "episodes": len(rows),
            "case_conditions": len({r["case_id"] for r in rows}),
            "task_families": len({(r["workflow"], r["variant_id"]) for r in rows}),
            "workflows": sorted({r["workflow"] for r in rows}),
            "faults": sorted({r["fault"] for r in rows}),
            "agents": sorted({r["agent_id"] for r in rows}),
            "agent_kinds": sorted({r["agent_kind"] for r in rows}),
            "run_status_counts": dict(Counter(r.get("run_status", "scripted_completed") for r in rows)),
            "metrics": metrics, "seed_consistency": seed_consistency(rows)}


def _safe_data(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")


def render_report(summary: dict, traces: list[dict], manifest: dict, output: Path) -> None:
    """Render exact tables and interactive trace filters; no remote assets."""
    totals = [m for m in summary["metrics"] if m["workflow"] == "all" and m["fault"] == "all"]
    consistency = {c["agent_id"]: c for c in summary.get("seed_consistency", []) if c["fault"] == "all"}

    def pass_k(agent: str) -> str:
        c = consistency.get(agent)
        return f"{c['all_seeds_success_rate']:.1%} (k={c['k']})" if c and c["k"] >= 2 else "—"

    table = "".join("<tr>" + f"<td>{html.escape(m['agent_id'])}</td><td>{html.escape(m['agent_kind'])}</td><td>{m['episodes']}</td>" +
                    f"<td>{m['success_rate']:.1%}</td><td>{pass_k(m['agent_id'])}</td><td>{m['duplicate_episode_rate']:.1%}</td>" +
                    f"<td>{m['missing_episode_rate']:.1%}</td><td>{m['false_success_rate']:.1%}</td>" +
                    f"<td>{m['mean_tool_calls']:.2f}</td></tr>" for m in totals)
    kinds = ", ".join(summary["agent_kinds"])
    payload = _safe_data({"summary": summary, "traces": traces})
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Recovery Eval · Run Report</title><style>
body{margin:0;background:#f2f5f8;color:#193147;font:15px/1.7 system-ui,sans-serif}main{max-width:1250px;margin:auto;padding:38px 24px}.top{font-size:12px;color:#087c79;letter-spacing:2px}h1{font-size:36px;margin:10px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.card,.notice,details{background:white;border:1px solid #dbe4eb;border-radius:12px;padding:16px}.card strong{display:block;font-size:30px}.notice{border-left:5px solid #138a83}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{padding:10px;border-bottom:1px solid #dce5ed;text-align:left}th{background:#e5edf4}h2{margin-top:32px}.scroll{overflow:auto}select,input{font:inherit;padding:8px;border:1px solid #bed1dd;border-radius:7px;background:white;margin:4px}.muted{color:#5e7083;font-size:13px}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.55 ui-monospace,monospace;background:#f3f6f8;padding:12px;max-height:440px;overflow:auto}details{margin:9px 0}summary{cursor:pointer}.fail{color:#a7442e}.pass{color:#087c79}@media(max-width:700px){main{padding:22px 12px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}}</style>
<main><div class="top">AGENT RECOVERY EVAL / V0.2</div><h1>多工作流执行与故障恢复评测</h1>
<p>固定公开步骤计划、业务参数和故障条件，以最终状态和提交账本判断结果。</p>
<div class="cards"><div class="card"><strong>__EPISODES__</strong>执行回合</div><div class="card"><strong>__WORKFLOWS__</strong>工作流</div><div class="card"><strong>__FAMILIES__</strong>业务参数族</div><div class="card"><strong>__FAULTS__</strong>条件（含对照）</div></div>
<div class="notice">本报告的执行类型：<strong>__KINDS__</strong>。规则策略与真实模型分别标记；模型的格式错误、截断、工具错误和基础设施错误保留在原始轨迹中。公开计划意味着本任务测执行与恢复，不测隐藏流程规划。云端任务：__CLOUD__。</div>
<h2>1. 总体结果</h2><div class="scroll"><table><tr><th>Agent / 策略</th><th>执行类型</th><th>回合</th><th>完成率</th><th>跨种子全通过 pass^k</th><th>重复回合率</th><th>漏做回合率</th><th>错误宣称成功率</th><th>平均调用</th></tr>__TABLE__</table></div><p class="muted">pass^k：同一工作流、业务变体与条件的全部 k 个种子都成功才算通过。种子改变故障落在哪一步及迟到／陈旧读调度，所以它比逐回合完成率更能暴露“只在部分步骤上能恢复”的策略；少于 2 个种子时显示 —。</p>
<h2>2. 按工作流与故障查看</h2><label>Agent <select id="agent"></select></label><label>工作流 <select id="workflow"></select></label><label>条件 <select id="fault"></select></label><div id="slice" class="notice"></div>
<h2>3. 可检查的原始轨迹样例</h2><p class="muted">报告内展示每个 Agent／工作流／条件下的代表样例；全部回合在 episodes.jsonl。失败优先，包含原始模型输出及工具返回。</p><input id="search" placeholder="搜索 case ID、状态或故障"><div id="traces"></div>
<h2>4. 复现与限制</h2><p>生成器参数、模型 revision、预算、代码 SHA256、run 状态和文件校验保存在 manifest.json。条件配对、bootstrap 与放行决定由独立 compare 命令生成。本报告不给多业务总体泛化或新算法结论；模拟环境的幂等契约和永久故障边界必须一并解释。</p>
<p class="muted">__TIME__ · 此 HTML 不依赖外部脚本、字体或账户。</p></main><script id="data" type="application/json">__DATA__</script>
<script>const D=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);function opts(id,items){for(const x of ['all',...items]){const o=document.createElement('option');o.value=x;o.textContent=x;$(id).appendChild(o)}}opts('agent',D.summary.agents);opts('workflow',D.summary.workflows);opts('fault',D.summary.faults);$('agent').value=D.summary.agents[0];function render(){const a=$('agent').value,w=$('workflow').value,f=$('fault').value;const m=D.summary.metrics.find(x=>x.agent_id===a&&x.workflow===w&&x.fault===f);$('slice').textContent=m?`回合 ${m.episodes} · 完成 ${(m.success_rate*100).toFixed(1)}% · 重复 ${(m.duplicate_episode_rate*100).toFixed(1)}% · 漏做 ${(m.missing_episode_rate*100).toFixed(1)}% · 模型响应无效 ${m.invalid_model_responses} · 截断 ${m.truncated_model_responses} · 基础设施错误回合 ${m.infrastructure_error_episodes}`:'请选择具体 Agent。';const q=$('search').value.toLowerCase();$('traces').replaceChildren();for(const r of D.traces.filter(r=>(a==='all'||r.agent_id===a)&&(w==='all'||r.workflow===w)&&(f==='all'||r.fault===f)&&JSON.stringify(r).toLowerCase().includes(q)).slice(0,30)){const e=document.createElement('details'),s=document.createElement('summary'),p=document.createElement('pre');s.textContent=`${r.case_id} · ${r.agent_id} · ${r.success?'通过':'失败'} · ${r.run_status||'scripted'}`;s.className=r.success?'pass':'fail';p.textContent=JSON.stringify(r,null,2);e.append(s,p);$('traces').appendChild(e)}}for(const id of ['agent','workflow','fault','search'])$(id).addEventListener(id==='search'?'input':'change',render);render();</script></html>"""
    replacements = {"__EPISODES__": f"{summary['episodes']:,}", "__WORKFLOWS__": str(len(summary["workflows"])),
                    "__FAMILIES__": str(summary["task_families"]), "__FAULTS__": str(len(summary["faults"])),
                    "__KINDS__": html.escape(kinds), "__CLOUD__": "已执行" if manifest.get("cloud_execution") else "未执行",
                    "__TABLE__": table, "__TIME__": html.escape(manifest["created_utc"]), "__DATA__": payload}
    for token, value in replacements.items():
        page = page.replace(token, value)
    output.write_text(page, encoding="utf-8")


def finalize_run(rows: list[dict], out: str | Path, metadata: dict | None = None) -> dict:
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    protected = ("episodes.jsonl", "summary.json", "example_traces.json", "metrics.csv", "report.html")
    if any((directory / name).exists() for name in protected):
        raise FileExistsError("Run artifacts already exist; use a fresh directory to preserve evidence")
    with (directory / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    end_hashes = source_hashes()
    start_hashes = (metadata or {}).get("code_sha256", end_hashes)
    changed_sources = sorted(name for name in set(start_hashes) | set(end_hashes)
                             if start_hashes.get(name) != end_hashes.get(name))
    manifest = {"schema_version": "2.0", "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "backend": "local", "cloud_execution": False, "code_sha256": start_hashes, **(metadata or {}),
                "code_sha256_end": end_hashes, "source_changed_during_run": bool(changed_sources),
                "changed_source_files": changed_sources}
    selected = {}
    for row in rows:
        key = (row["agent_id"], row["workflow"], row["fault"])
        if key not in selected or (selected[key]["success"] and not row["success"]):
            selected[key] = row
    examples = list(selected.values())
    write_json(directory / "summary.json", summary)
    write_json(directory / "example_traces.json", examples)
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["metrics"][0]))
        writer.writeheader(); writer.writerows(summary["metrics"])
    render_report(summary, examples, manifest, directory / "report.html")
    manifest["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(directory.iterdir()) if p.is_file() and p.name != "manifest.json"}
    write_json(directory / "manifest.json", manifest)
    return summary
