"""Run bounded, ephemeral CPU batches on Modal and save returned results locally.

This entrypoint never invokes an LLM. The recovery policies are deterministic
rule-based baselines. Importing this module registers the app but starts no job.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import modal


POLICIES = ("bounded_retry", "verify_then_retry", "idempotent_retry")
BATCH_SIZE = 12
app = modal.App("recovery-eval-lite")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_python_source("recovery_eval")
)


@app.function(
    image=image,
    cpu=(0.25, 0.5),
    memory=(256, 512),
    min_containers=0,
    max_containers=2,
    timeout=120,
    retries=0,
)
def evaluate_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate independent case/policy pairs in one remote CPU invocation."""
    from recovery_eval.core import run_case

    return [run_case(item["case"], item["policy"]) for item in items]


@app.local_entrypoint()
def main(
    tasks: int = 12,
    seeds: int = 5,
    limit: int = 60,
    out: str = "results/modal",
) -> None:
    """Run cloud batches; limit counts policy runs, with zero meaning all runs."""
    from recovery_eval.core import build_cases
    from recovery_eval.reporting import write_artifacts

    if limit < 0 or 0 < limit < len(POLICIES):
        raise ValueError("limit must be 0 (full suite) or at least 3")

    cases = build_cases(task_count=tasks, seeds=seeds)
    full_case_count = len(cases)
    # Keep matched policy comparisons even when the smoke-run cap is not a multiple of three.
    if limit:
        cases = cases[: limit // len(POLICIES)]
    items = [{"case": case, "policy": policy} for case in cases for policy in POLICIES]
    batches = [items[index : index + BATCH_SIZE] for index in range(0, len(items), BATCH_SIZE)]
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    print(f"Running {len(items)} policy runs in {len(batches)} Modal CPU batches.")

    # Artifacts are written only after every remote invocation has returned successfully.
    returned_batches = list(evaluate_batch.map(batches))
    results = [result for batch in returned_batches for result in batch]
    if len(results) != len(items):
        raise RuntimeError("Modal returned an unexpected result count; no report was written")

    metadata = {
        "backend": "modal",
        "cloud_execution": True,
        "execution_status": "completed",
        "policy_type": "rule_based_baselines",
        "llm_calls": 0,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_wall_seconds": time.perf_counter() - started,
        "modal_app_id": app.app_id,
        "modal_dashboard_url": app.get_dashboard_url(),
        "modal_sdk_version": modal.__version__,
        "requested_tasks": tasks,
        "requested_seeds": seeds,
        "requested_limit": limit,
        "available_cases": full_case_count,
        "evaluated_cases": len(cases),
        "evaluated_policy_runs": len(results),
        "policies": list(POLICIES),
        "selection": "full_suite" if len(cases) == full_case_count else "prefix_smoke_test",
        "batch_size": BATCH_SIZE,
        "batch_count": len(batches),
        "max_containers": 2,
        "function_timeout_seconds": 120,
    }
    write_artifacts(results, out, metadata=metadata)
    print(f"Saved {len(results)} returned cloud results to {out}")

