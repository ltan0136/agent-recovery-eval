"""Command-line entry point for credential-free experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .core import POLICIES, build_cases, run_case, run_suite
from .reporting import diagnosis_controls, diagnosis_rows, write_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible agent recovery evaluation; no keys required for demo.")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run scripted policy baselines, not language models")
    demo.add_argument("--tasks", type=int, default=12)
    demo.add_argument("--seeds", type=int, default=5)
    demo.add_argument("--out", default="results/local")
    compare = sub.add_parser("compare", help="Compare paired episodes under identical frozen conditions")
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--before-policy", required=True)
    compare.add_argument("--after-policy", required=True)
    compare.add_argument("--out", default="results/regression")
    replay = sub.add_parser("replay", help="Re-execute one deterministic simulator case")
    replay.add_argument("--case-id", required=True)
    replay.add_argument("--policy", choices=POLICIES, required=True)
    replay.add_argument("--tasks", type=int, default=12)
    replay.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if args.command == "compare":
        from .comparison import compare_files
        report = compare_files(args.before, args.after, args.before_policy, args.after_policy, args.out)
        print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, ensure_ascii=False))
        return
    cases = build_cases(args.tasks, args.seeds)
    if args.command == "replay":
        matches = [case for case in cases if case["case_id"] == args.case_id]
        if not matches:
            parser.error("case ID not found in the specified task/seed range")
        print(json.dumps(run_case(matches[0], args.policy), ensure_ascii=False, indent=2))
        return
    results = run_suite(cases)
    summary = write_artifacts(results, args.out)
    data_path = Path("integrations/prime_taskset/data/diagnosis_cases.jsonl")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = diagnosis_rows(results + diagnosis_controls(cases))
    # Select unique observable cases, not repeated prompt copies.
    unique: dict[str, dict] = {}
    for row in diagnostics:
        fingerprint = json.dumps({"request": row["request"], "trace": row["trace"]}, sort_keys=True)
        unique.setdefault(fingerprint, row)
    groups: dict[str, list[dict]] = {}
    for row in unique.values():
        label = json.dumps(row["expected"], sort_keys=True)
        groups.setdefault(label, []).append(row)
    balanced = []
    for index in range(max(map(len, groups.values()))):
        for label in sorted(groups):
            if index < len(groups[label]):
                balanced.append(groups[label][index])
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in balanced), encoding="utf-8")
    print(json.dumps({"episodes": summary["episodes"], "report": str(Path(args.out) / "report.html"),
                      "diagnosis_cases": len(balanced), "mode": "scripted_policy_simulation"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
