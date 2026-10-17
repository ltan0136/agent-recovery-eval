"""Rebuild the v0.2 settled-evidence smoke fixture from the trusted core engine."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def build_fixture(out: Path) -> int:
    from recovery_eval.cli import main
    from recovery_eval.engine import Environment
    from recovery_eval.workflows import build_matrix

    if out.exists():
        raise ValueError("Output already exists; use a fresh path")
    with TemporaryDirectory(prefix="recovery-prime-fixture-") as temporary:
        run_dir = Path(temporary) / "source-run"
        main(["run", "--variants", "1", "--seeds", "1",
              "--faults", "clean,lost_ack,permanent_failure",
              "--policies", "no_retry,bounded_retry,idempotent_retry", "--out", str(run_dir)])
        source = run_dir / "episodes.jsonl"
        changed_keys = {"ticket_dispatch": "title", "inventory_order": "quantity",
                        "meeting_booking": "room", "document_publication": "content"}
        with source.open("a", encoding="utf-8") as stream:
            for case in build_matrix(variants=1, seeds=1, faults="clean"):
                environment = Environment(case)
                for index, step in enumerate(environment.public_task["steps"]):
                    arguments = deepcopy(step["arguments"])
                    if index == 0:
                        key = changed_keys[case["workflow"]]
                        value = arguments[key]
                        arguments[key] = value + 1 if type(value) is int else "Incorrect value for diagnosis control"
                    response = environment.invoke(step["tool"], arguments)
                    if not response["ok"]:
                        raise RuntimeError("Known wrong-payload control must still commit every step")
                row = environment.finish(True, agent_id="known_wrong_payload", agent_kind="known_answer_control")
                if row["success"] or row["duplicates"] != 0 or row["missing"] != 0:
                    raise RuntimeError("Known control must fail with zero duplicate and missing counts")
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        main(["export-diagnosis", "--episodes", str(source), "--out", str(out)])
    return 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = build_fixture(args.out)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"fixture_cases": count, "source": "trusted_v0.2_core", "cloud_execution": False}))


if __name__ == "__main__":
    main()
