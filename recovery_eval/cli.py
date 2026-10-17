"""Public v2 command line: matrices, model-in-the-loop runs, replay and comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

from .artifacts import finalize_run, read_jsonl, source_hashes, write_json
from .engine import run_scripted
from .workflows import FAULTS, POLICIES, build_matrix, workflow_specs


def _csv(value: str | None):
    return [part.strip() for part in value.split(",") if part.strip()] if value else None


def _base_arguments(parser):
    parser.add_argument("--workflows", help="Comma-separated workflow IDs; default all")
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--faults", help="Comma-separated conditions; default all eight")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="recovery-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show workflow contracts and supported conditions")
    matrix = sub.add_parser("matrix", help="Export a frozen, hash-addressed case matrix")
    _base_arguments(matrix); matrix.add_argument("--out", default="runs/cases.jsonl")
    run = sub.add_parser("run", help="Run deterministic scripted policies without credentials")
    _base_arguments(run); run.add_argument("--policies", default=",".join(POLICIES))
    run.add_argument("--max-tool-calls", type=int, default=24)
    run.add_argument("--out", default="runs/scripted")
    model = sub.add_parser("model", help="Run real models that execute tools in the simulator")
    _base_arguments(model); model.set_defaults(variants=1, seeds=1, faults="clean,lost_ack")
    model.add_argument("--backend", choices=("local", "api"), default="local")
    model.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    model.add_argument("--revision", default="main")
    model.add_argument("--cache-dir")
    model.add_argument("--allow-download", action="store_true")
    model.add_argument("--threads", type=int, default=4)
    model.add_argument("--max-new-tokens", type=int, default=160)
    model.add_argument("--max-turns", type=int, default=8)
    model.add_argument("--max-tool-calls", type=int, default=12)
    model.add_argument("--format-retries", type=int, choices=(0, 1), default=1)
    model.add_argument("--prompts", default="basic,guarded")
    model.add_argument("--base-url")
    model.add_argument("--api-key-env", default="PRIME_API_KEY")
    model.add_argument("--out", default="runs/model")
    replay = sub.add_parser("replay", help="Re-execute the saved case and scripted policy")
    replay.add_argument("--episodes", required=True); replay.add_argument("--case-id", required=True)
    replay.add_argument("--policy", choices=POLICIES, default="idempotent_retry")
    compare = sub.add_parser("compare", help="Paired regression and safety checks")
    compare.add_argument("--before", required=True); compare.add_argument("--after", required=True)
    compare.add_argument("--before-agent", required=True); compare.add_argument("--after-agent", required=True)
    compare.add_argument("--max-allowed-drop", type=float, default=0.0)
    compare.add_argument("--out", default="runs/comparison.json")
    compare.add_argument("--fail-on-block", action="store_true")
    export = sub.add_parser("export-diagnosis", help="Export settled evidence for the separate Prime diagnosis task")
    export.add_argument("--episodes", required=True); export.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps({"workflows": {key: {"title": spec.title, "steps": [s.name for s in spec.steps]}
                                         for key, spec in workflow_specs().items()},
                          "conditions": FAULTS, "scripted_policies": POLICIES}, ensure_ascii=False, indent=2)); return
    if args.command == "replay":
        found = [r for r in read_jsonl(args.episodes) if r["case_id"] == args.case_id]
        if not found:
            parser.error("Case ID absent")
        policy_rows = [r for r in found if r.get("agent_id") == args.policy]
        candidates = policy_rows or found
        if len(policy_rows) > 1:
            parser.error("More than one episode matches the case and policy; use a unique evidence file")
        if len(candidates) > 1 and len({json.dumps(r["execution_config"], sort_keys=True) for r in candidates}) > 1:
            parser.error("Case has multiple execution configurations; choose an existing policy or a unique evidence file")
        saved = candidates[0]
        result = run_scripted(saved["case_config"], args.policy,
                              max_calls=saved["execution_config"]["max_calls"],
                              idempotency=saved["execution_config"]["idempotency_supported"])
        if not saved.get("condition_hash") or result["condition_hash"] != saved["condition_hash"]:
            parser.error("Saved execution conditions are missing or differ from the replay; refusing a misleading reproduction")
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    if args.command == "compare":
        if Path(args.out).exists():
            parser.error("Output file already exists; choose a fresh path to preserve evidence")
        from .regression import compare_runs
        result = compare_runs(read_jsonl(args.before), read_jsonl(args.after), args.before_agent,
                              args.after_agent, max_allowed_drop=args.max_allowed_drop)
        write_json(Path(args.out), result)
        print(json.dumps({"decision": result["decision"], "summary": result["summary"]}, ensure_ascii=False))
        if args.fail_on_block and result["decision"] == "block":
            raise SystemExit(2)
        return
    if args.command == "export-diagnosis":
        if Path(args.out).exists():
            parser.error("Output file already exists; choose a fresh path to preserve evidence")
        rows = read_jsonl(args.episodes)
        destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            for r in rows:
                diagnostic = {"id": f"{r['case_id']}:{r['agent_id']}", "request": r["public_task"],
                              "trace": [{"tool": "inspect_settled_state", "result": r["final_state"]},
                                        {"tool": "inspect_settled_ledger", "result": r["ledger"]}],
                              "expected": {"duplicates": r["duplicates"], "missing": r["missing"], "success": r["success"]}}
                handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
        print(json.dumps({"diagnosis_cases": len(rows), "scope": "settled evidence diagnosis; not recovery"})); return
    cases = build_matrix(_csv(args.workflows), args.variants, args.seeds, _csv(args.faults))
    if args.command == "matrix":
        path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            parser.error("Output file already exists; choose a fresh path to preserve evidence")
        with path.open("x", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
        print(json.dumps({"cases": len(cases), "out": str(path)})); return
    rows = []
    directory = Path(args.out); directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.jsonl"
    if any(directory.iterdir()):
        parser.error("Output directory is not empty; choose a fresh directory to preserve evidence")
    started = time.perf_counter()
    metadata = {"command": args.command, "requested_workflows": _csv(args.workflows),
                "variants": args.variants, "seeds": args.seeds, "conditions": _csv(args.faults),
                "case_count": len(cases), "status": "running", "cloud_execution": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "code_sha256": source_hashes(),
                "max_tool_calls": args.max_tool_calls}
    if args.command == "run":
        metadata["policies"] = _csv(args.policies)
    else:
        metadata.update({"requested_backend": args.backend, "requested_model": args.model,
                         "requested_revision": args.revision, "prompt_variants": _csv(args.prompts),
                         "max_turns": args.max_turns, "max_new_tokens": args.max_new_tokens,
                         "format_retries": args.format_retries, "allow_download": args.allow_download})
    write_json(directory / "manifest.json", metadata)
    try:
        if args.command == "run":
            policies = _csv(args.policies)
            if not policies or any(p not in POLICIES for p in policies) or len(set(policies)) != len(policies):
                parser.error("Policies must be distinct known policies")
            jobs = ((case, policy) for case in cases for policy in policies)
            for case, policy in jobs:
                row = run_scripted(case, policy, max_calls=args.max_tool_calls)
                rows.append(row)
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metadata["policies"] = policies
        else:
            from .agents import LocalTransformersAdapter, OpenAICompatibleAdapter, run_agent
            prompts = _csv(args.prompts)
            if not prompts or any(p not in ("basic", "guarded") for p in prompts) or len(set(prompts)) != len(prompts):
                parser.error("Prompts must be distinct basic/guarded variants")
            if args.backend == "local":
                adapter = LocalTransformersAdapter(args.model, revision=args.revision, cache_dir=args.cache_dir,
                    max_new_tokens=args.max_new_tokens, threads=args.threads, local_files_only=not args.allow_download)
            else:
                if not args.base_url:
                    parser.error("API backend requires --base-url")
                adapter = OpenAICompatibleAdapter(args.model, args.base_url, api_key_env=args.api_key_env,
                                                 max_new_tokens=args.max_new_tokens)
            metadata["model_backend"] = adapter.describe()
            write_json(directory / "manifest.json", metadata)
            for case in cases:
                for prompt in prompts:
                    row = run_agent(case, adapter, prompt_variant=prompt, max_turns=args.max_turns,
                                    max_tool_calls=args.max_tool_calls, format_retries=args.format_retries)
                    rows.append(row)
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps({"case_id": row["case_id"], "agent": row["agent_id"],
                                      "success": row["success"], "tool_calls": row["tool_calls"],
                                      "status": row["run_status"]}), flush=True)
            metadata.update({"model_backend": adapter.describe(), "prompt_variants": prompts,
                             "max_turns": args.max_turns, "max_tool_calls": args.max_tool_calls,
                             "format_retries": args.format_retries})
        metadata.update({"status": "completed", "duration_seconds": time.perf_counter() - started})
        summary = finalize_run(rows, directory, metadata)
        print(json.dumps({"episodes": summary["episodes"], "report": str(directory / "report.html")}, ensure_ascii=False))
    except BaseException as exc:
        metadata.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         "error_type": type(exc).__name__, "completed_episodes": len(rows),
                         "duration_seconds": time.perf_counter() - started})
        write_json(directory / "manifest.json", metadata)
        raise


if __name__ == "__main__":
    main()
