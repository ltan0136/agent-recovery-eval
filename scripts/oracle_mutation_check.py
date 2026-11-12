"""Check that the test suite notices when any single judge check is switched off.

This is fault injection for the judge itself. Each mutant disables exactly one
check in Environment._oracle() or Environment.finish() (recovery_eval/engine.py)
inside a temporary copy of the repository, then runs the full unittest suite
there. A mutant is "killed" when at least one test fails; it "survived" when
the whole suite still passes, which means no test depends on that check.

Only the v0.2 judge is covered: not the v0.1 core.py scorer, the regression
gate or the Prime scoring package. Each mutant is located by an exact source
snippet; if engine.py changes so that a snippet no longer matches, the mutant
is reported as "not_applied" and the list below must be updated. When you add a
judge check, add a mutant for it here and a test that kills it.

Exit status: 0 when every mutant is killed, 1 when any survives, 2 when any
cannot be applied, 3 when the unmodified suite already fails.
The working tree is never modified.

    python scripts/oracle_mutation_check.py
    python scripts/oracle_mutation_check.py --workers 2 --json runs/oracle_mutants.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path("recovery_eval") / "engine.py"
IGNORED = shutil.ignore_patterns(".git", "runs", "__pycache__", "*.egg-info", "build", "dist", ".venv", "venv")


def _drop(reason: str) -> tuple[str, str]:
    """Turn one business_violations.append(...) statement into a no-op."""
    return f'business_violations.append("{reason}")', "pass"


# (id, what is switched off, exact source snippet, replacement)
MUTANTS: list[tuple[str, str, str, str]] = [
    ("M01", "duplicate count forced to 0",
     "duplicates = sum(max(0, count - 1) for count in counts.values())", "duplicates = 0"),
    ("M02", "missing steps forced to empty",
     'missing_steps = [step["name"] for step in self._plan if counts[step["name"]] == 0]', "missing_steps = []"),
    ("M03", "wrong-payload count forced to 0",
     'wrong_payload = sum(effect["arguments"] != self._by_name[effect["step"]]["arguments"] for effect in self._ledger)',
     "wrong_payload = 0"),
    ("M04", "dependency-order violations never counted",
     'order_violations += sum(required not in seen for required in self._by_name[effect["step"]]["requires"])',
     "order_violations += 0"),
    ("M05", "state/ledger count mismatch check disabled",
     "if len(state[collection]) != counts[step_name]:", "if False:"),
    ("M06", "state content mismatch check disabled",
     "if any(any(record.get(key) != value for key, value in expected_args.items()) for record in state[collection]):",
     "if False:"),
    ("M07", "ticket final-state check disabled", *_drop("ticket_final_state_mismatch")),
    ("M08", "broken ticket reference check disabled", *_drop("broken_ticket_reference")),
    ("M09", "stock over-reservation check disabled",
     'resource_violations += int(state["available_stock"] < 0 or reserved > params["quantity"])',
     "resource_violations += 0"),
    ("M10", "over-charge check disabled",
     'resource_violations += int(charged > params["amount_cents"])', "resource_violations += 0"),
    ("M11", "inventory conservation check disabled", *_drop("inventory_conservation_failure")),
    ("M12", "order final-state check disabled", *_drop("order_final_state_mismatch")),
    ("M13", "shipment reference check disabled", *_drop("shipment_resource_reference_mismatch")),
    ("M14", "room over-capacity check disabled",
     'resource_violations += sum(max(0, count - state["room_capacity"]) for count in occupied.values())',
     "resource_violations += 0"),
    ("M15", "single invitation batch check disabled", *_drop("confirmation_requires_one_invitation_batch")),
    ("M16", "confirmation reference check disabled", *_drop("confirmation_resource_reference_mismatch")),
    ("M17", "publication-needs-matching-approval check disabled", *_drop("publication_without_matching_approval")),
    ("M18", "failed document check disabled", *_drop("failed_document_checks")),
    ("M19", "judge always reports success",
     'return {"success": not reasons, "passed": not reasons,', 'return {"success": True, "passed": True,'),
    ("M20", "false_success always False",
     '"false_success": claimed_success and not score["success"],', '"false_success": False,'),
    ("M21", "late writes not settled before grading",
     "self._advance(max(item[0] for item in self._schedule) - self._tick)", "pass"),
    ("M22", "duplicates counted but no longer fail the episode",
     '(duplicates, "duplicate_effects")', '(0, "duplicate_effects")'),
    ("M23", "missing steps counted but no longer fail the episode",
     '(len(missing_steps), "missing_steps")', '(0, "missing_steps")'),
    ("M24", "wrong payload counted but no longer fails the episode",
     '(wrong_payload, "wrong_payload")', '(0, "wrong_payload")'),
]


def run_suite(directory: Path) -> tuple[bool, list[str], str]:
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                             cwd=directory, capture_output=True, text=True, timeout=900)
    output = process.stderr + process.stdout
    failing = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+)", output, flags=re.MULTILINE)))
    lines = output.strip().splitlines()
    return process.returncode == 0, failing, lines[-1] if lines else ""


def run_mutant(mutant: tuple[str, str, str, str], workspace: Path) -> dict:
    ident, description, old, new = mutant
    copy = workspace / ident
    shutil.copytree(ROOT, copy, ignore=IGNORED)
    path = copy / TARGET
    source = path.read_text(encoding="utf-8")
    occurrences = source.count(old)
    if occurrences != 1:
        shutil.rmtree(copy, ignore_errors=True)
        return {"id": ident, "description": description, "status": "not_applied", "occurrences": occurrences}
    path.write_text(source.replace(old, new), encoding="utf-8")
    passed, failing, _ = run_suite(copy)
    shutil.rmtree(copy, ignore_errors=True)
    return {"id": ident, "description": description, "status": "survived" if passed else "killed",
            "failing_tests": len(failing), "examples": failing[:3]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--json", type=Path, help="Also write the per-mutant results to this file.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    with tempfile.TemporaryDirectory(prefix="oracle-mutants-") as temporary:
        workspace = Path(temporary)
        baseline = workspace / "baseline"
        shutil.copytree(ROOT, baseline, ignore=IGNORED)
        passed, failing, tail = run_suite(baseline)
        if not passed:
            print(f"The unmodified suite fails ({tail}); fix it before checking mutants: {failing[:5]}")
            return 3
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda mutant: run_mutant(mutant, workspace), MUTANTS))
    for row in results:
        detail = f"  e.g. {', '.join(row['examples'][:2])}" if row["status"] == "killed" else ""
        print(f"{row['id']}  {row['status']:<11}  {row['description']}{detail}")
    counts = {status: sum(row["status"] == status for row in results)
              for status in ("killed", "survived", "not_applied")}
    print(json.dumps({"mutants": len(results), **counts}))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if counts["not_applied"]:
        return 2
    return 1 if counts["survived"] else 0


if __name__ == "__main__":
    sys.exit(main())
