"""Build a consolidated review HTML and compact Git-tracked evidence snapshot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_eval.artifacts import read_jsonl, render_report, summarize, write_json
from recovery_eval.regression import compare_runs
from recovery_eval.workflows import FAULTS, POLICIES, workflow_specs
from scripts.reproduce_release import MODELS
from scripts.verify_artifacts import verify


def validate_protocol(name: str, batch: list[dict], manifest: dict) -> None:
    """Reject substituted run types or selectively shortened release matrices."""
    expected_workflows = set(workflow_specs())
    if {r["workflow"] for r in batch} != expected_workflows:
        raise ValueError(f"Release workflow set differs: {name}")
    if name == "scripted":
        if {r["agent_kind"] for r in batch} != {"scripted"}:
            raise ValueError("The scripted release includes non-scripted episodes")
        if {r["agent_id"] for r in batch} != set(POLICIES) or {r["fault"] for r in batch} != set(FAULTS):
            raise ValueError("Scripted release policies or conditions differ")
        expected_cells = {(workflow, fault, policy, variant, seed)
                          for workflow in expected_workflows for fault in FAULTS
                          for policy in POLICIES for variant in range(12) for seed in range(4)}
        actual_cells = {(r["workflow"], r["fault"], r["agent_id"], r["variant"], r["seed"]) for r in batch}
        if actual_cells != expected_cells:
            raise ValueError("The scripted matrix has missing or substituted cells")
        if any(r["execution_config"]["max_calls"] != 24 for r in batch):
            raise ValueError("Scripted tool-call budget differs from release protocol")
        return
    models = {key: (model, revision) for key, model, revision in MODELS}
    model, revision = models[name]
    if manifest.get("requested_model") != model or manifest.get("requested_revision") != revision:
        raise ValueError(f"Requested model identity or revision differs: {name}")
    expected_cells = {(workflow, fault, prompt) for workflow in expected_workflows
                      for fault in ("clean", "lost_ack") for prompt in ("basic", "guarded")}
    actual_cells = {(r["workflow"], r["fault"], r["prompt_variant"]) for r in batch}
    if actual_cells != expected_cells:
        raise ValueError(f"The model matrix has missing or substituted cells: {name}")
    for row in batch:
        backend = row["model_backend"]
        if row["agent_kind"] != "local_model" or row["model"] != model:
            raise ValueError(f"The model release contains a substituted agent: {name}")
        if backend.get("requested_revision") != revision or backend.get("resolved_revision_sha") not in (None, revision):
            raise ValueError(f"Model revision differs from the lock: {name}")
        generated = any(call.get("generated_token_ids") is not None for call in row.get("model_calls", []))
        if generated and backend.get("resolved_revision_sha") != revision:
            raise ValueError(f"Generated output has no verified resolved revision: {name}")
        if (row.get("max_turns") != 8 or row.get("max_tool_calls") != 12
                or row.get("format_retry_limit") != 1 or backend.get("max_new_tokens") != 160
                or backend.get("threads") != 4 or backend.get("dtype") != "float32"
                or backend.get("device") != "cpu" or backend.get("do_sample") is not False
                or row["seed"] != 0 or row["variant"] != 0):
            raise ValueError(f"Model execution protocol differs: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted", type=Path, required=True)
    parser.add_argument("--smol", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    rows, traces, runs = [], [], {}
    for name, directory, count in (("scripted", args.scripted, 6144), ("smol", args.smol, 16), ("qwen", args.qwen, 16)):
        manifest = json.loads((directory / "manifest.json").read_text())
        try:
            verify(directory)
        except (ValueError, KeyError, OSError) as error:
            raise SystemExit(f"Invalid run artifacts: {directory}: {error}") from error
        batch = read_jsonl(directory / "episodes.jsonl")
        if len(batch) != count:
            raise SystemExit(f"Unexpected number of episodes: {directory}")
        summary = summarize(batch)
        try:
            validate_protocol(name, batch, manifest)
        except (ValueError, KeyError) as error:
            raise SystemExit(f"Invalid release protocol: {directory}: {error}") from error
        rows.extend(batch)
        traces.extend(json.loads((directory / "example_traces.json").read_text()))
        runs[name] = {"manifest": manifest, "summary": summary}
    comparisons = {}
    for name in ("unchanged", "intentional-regression"):
        source = args.comparisons / f"{name}.json"
        result = json.loads(source.read_text())
        expected = compare_runs(rows[:6144], rows[:6144], before_agent="idempotent_retry",
                                after_agent="idempotent_retry" if name == "unchanged" else "bounded_retry")
        if result != expected:
            raise SystemExit(f"Comparison differs from the complete raw scripted run: {source}")
        comparisons[name] = {"decision": result["decision"], "summary": {
            key: value for key, value in result["summary"].items()
            if key not in ("regression_case_ids", "improvement_case_ids")},
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if comparisons["unchanged"]["decision"] != "pass" or comparisons["intentional-regression"]["decision"] != "block":
        raise SystemExit("Unexpected regression gate results")
    model_analysis = {}
    grouped = defaultdict(list)
    for row in rows:
        if row["agent_kind"] == "local_model":
            grouped[row["agent_id"]].append(row)
    for agent, batch in sorted(grouped.items()):
        clean = [r for r in batch if r["fault"] == "clean"]
        faulty = [r for r in batch if r["fault"] != "clean"]
        clean_success = {(r["workflow"], r["variant_id"], r["seed"]) for r in clean if r["success"]}
        eligible = [r for r in faulty if (r["workflow"], r["variant_id"], r["seed"]) in clean_success]
        calls = [c for r in batch for c in r.get("model_calls", [])]
        tool_names = Counter(e["tool"] for r in batch for e in r["trace"] if e["event"] == "tool_call")
        model_analysis[agent] = {
            "episodes": len(batch), "clean_episodes": len(clean), "clean_successes": sum(r["success"] for r in clean),
            "fault_label_episodes": len(faulty), "fault_triggered_episodes": sum(
                any(e["event"] == "fault_triggered" for e in r["trace"]) for r in faulty),
            "tool_calls_by_name": dict(tool_names), "model_calls": len(calls),
            "tool_errors_by_type": dict(Counter(
                c["tool_response"]["error"]["type"] for c in calls
                if c.get("tool_response", {}).get("ok") is False)),
            "committed_effects": sum(len(r["ledger"]) for r in batch),
            "partial_effect_episodes": sum(bool(r["ledger"]) and not r["success"] for r in batch),
            "invalid_responses": sum(c["status"] == "invalid_action" for c in calls),
            "truncated_responses": sum(bool(c.get("metadata", {}).get("generation_truncated")) for c in calls),
            "conditional_recovery_eligible": len(eligible),
            "conditional_recovery_success_rate": sum(r["success"] for r in eligible) / len(eligible) if eligible else None,
            "note": "Descriptive diagnosis only; no recovery-rate estimate when paired clean success is absent.",
        }
    model_episodes = sum(r["agent_kind"] == "local_model" for r in rows)
    scripted_episodes = sum(r["agent_kind"] == "scripted" for r in rows)
    snapshot = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": runs,
                "comparisons": comparisons, "real_model_analysis": model_analysis,
                "real_model_episodes": model_episodes, "scripted_episodes": scripted_episodes,
                "cloud_execution": False, "hosted_github_ci_executed": False}
    write_json(args.snapshot, snapshot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_report(summarize(rows), traces, {"created_utc": snapshot["created_utc"], "cloud_execution": False}, args.out)
    page = args.out.read_text()
    regression = comparisons["intentional-regression"]["summary"]
    notice = (f'<div class="notice"><strong>审查快照：</strong>{scripted_episodes:,} 回合规则基线与{model_episodes}回合真实本地模型。'
              '规则每回合24次工具预算；模型为8次请求、12次工具、160个新token。'
              '两类成绩用于不同目的，不能据此做同预算排名。'
              f'故意策略变更的{regression["paired_cases"]:,}对比较检出{regression["regressions"]}个退化，门禁block；相同策略自比较pass。'
              '模型只有在调用目标写工具时才实际触发故障；配对clean未成功时，恢复能力无法单独估计。'
              '完整代码、PR说明和原始JSONL见配套审查包与实验记录包；本HTML可离线筛选。</div>')
    page = page.replace('<h2>1. 总体结果</h2>', notice + '<h2>1. 总体结果</h2>')
    args.out.write_text(page)
    print(json.dumps({"out": str(args.out), "snapshot": str(args.snapshot), "episodes": len(rows)}))


if __name__ == "__main__":
    main()
