"""Optional bounded Modal CPU execution for the v0.2 multi-workflow engine.

Importing this module only registers SDK objects. It starts no cloud work, loads
no model, and sends no inference request. Cloud runs require an explicit
``modal run`` command. Function.local() is available for SDK-aware local checks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import modal


BATCH_SIZE = 16
app = modal.App("agent-recovery-eval")
base_image = modal.Image.debian_slim(python_version="3.11")
image = base_image.add_local_python_source("recovery_eval")
model_image = (
    base_image
    .pip_install("torch==2.14.1+cpu", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.20.1", "huggingface_hub==2.2.0", "socksio==1.0.0")
    .add_local_python_source("recovery_eval")
)


@app.function(image=image, cpu=(0.25, 0.5), memory=(256, 512),
              min_containers=0, max_containers=2, timeout=120, retries=0)
def evaluate_batch(items: list[dict[str, Any]], max_calls: int = 24) -> list[dict[str, Any]]:
    """Run independent v0.2 scripted baseline episodes, not language models."""
    from recovery_eval.engine import run_scripted

    return [run_scripted(item["case"], item["policy"], max_calls=max_calls) for item in items]


@app.function(image=model_image, cpu=2, memory=2048,
              min_containers=0, max_containers=1, timeout=900, retries=0)
def evaluate_model_batch(
    cases: list[dict[str, Any]], model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    revision: str = "main", prompts: tuple[str, ...] = ("basic", "guarded"),
    max_turns: int = 8, max_calls: int = 12, max_new_tokens: int = 160,
    threads: int = 2, cache_dir: str | None = None, allow_download: bool = False,
) -> list[dict[str, Any]]:
    """Load one real CPU model and let it execute tools in the v0.2 simulator.

    A local .local() invocation uses the caller's installed model dependencies
    and optional cache. Remote invocations have ephemeral storage; downloading
    public model weights requires allow_download=True. No API key is needed.
    """
    from recovery_eval.agents import LocalTransformersAdapter, run_agent

    if not cases or len(cases) * len(prompts) > 16:
        raise ValueError("Use between 1 and 16 model episodes per invocation")
    if not prompts or len(set(prompts)) != len(prompts) or any(p not in ("basic", "guarded") for p in prompts):
        raise ValueError("prompts must contain distinct basic/guarded variants")
    adapter = LocalTransformersAdapter(model, revision=revision, threads=threads,
        cache_dir=cache_dir, max_new_tokens=max_new_tokens, local_files_only=not allow_download)
    return [run_agent(case, adapter, prompt_variant=prompt, max_turns=max_turns,
                      max_tool_calls=max_calls, format_retries=1)
            for case in cases for prompt in prompts]


def _names(value: str) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def select_cases(workflows: str = "", faults: str = "", variants: int = 4,
                 seeds: int = 3, limit: int = 16, comparisons: int = 4) -> tuple[list[dict[str, Any]], int]:
    """Keep complete per-case comparison groups when applying a smoke-run cap."""
    from recovery_eval.workflows import build_matrix

    if type(comparisons) is not int or comparisons < 1:
        raise ValueError("comparisons must be a positive integer")
    if type(limit) is not int or limit < 0 or (limit and limit < comparisons):
        raise ValueError("limit must be 0 (full suite) or at least the comparison count")
    cases = build_matrix(_names(workflows), variants=variants, seeds=seeds, faults=_names(faults))
    available = len(cases)
    if limit:
        cases = cases[:limit // comparisons]
    return cases, available


@app.local_entrypoint()
def main(
    mode: str = "scripted", workflows: str = "", faults: str = "", variants: int = 4,
    seeds: int = 3, limit: int = 16, out: str = "runs/modal", max_calls: int = 24,
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct", revision: str = "main",
    prompts: str = "basic,guarded", max_turns: int = 8, max_new_tokens: int = 160,
    allow_model_download: bool = False,
) -> None:
    """Explicitly execute ephemeral cloud work; preserve complete comparison groups."""
    from recovery_eval.artifacts import finalize_run, source_hashes
    from recovery_eval.workflows import POLICIES

    if mode not in ("scripted", "local-model"):
        raise ValueError("mode must be scripted or local-model")
    prompt_names = _names(prompts) or []
    if mode == "local-model" and (not prompt_names or len(set(prompt_names)) != len(prompt_names)
            or any(name not in ("basic", "guarded") for name in prompt_names)):
        raise ValueError("prompts must contain distinct basic/guarded variants")
    comparisons = len(POLICIES) if mode == "scripted" else len(prompt_names)
    cases, available = select_cases(workflows, faults, variants, seeds, limit, comparisons)
    episode_count = len(cases) * comparisons
    if mode == "local-model" and (not limit or episode_count > 16):
        raise ValueError("local-model mode requires a nonzero limit of at most 16 episodes")
    if mode == "local-model" and not allow_model_download:
        raise ValueError("Remote CPU model runs need --allow-model-download for ephemeral weight downloads")
    directory = Path(out)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Output directory is not empty; choose a fresh directory")
    if type(max_calls) is not int or not 1 <= max_calls <= 1000:
        raise ValueError("max_calls must be between 1 and 1000")
    started_at = datetime.now(timezone.utc).isoformat()
    code_sha256 = source_hashes()
    started = time.perf_counter()
    print(f"Running {episode_count} {mode} episodes on Modal CPU.")
    if mode == "scripted":
        items = [{"case": case, "policy": policy} for case in cases for policy in POLICIES]
        batches = [items[index:index + BATCH_SIZE] for index in range(0, len(items), BATCH_SIZE)]
        returned = list(evaluate_batch.starmap((batch, max_calls) for batch in batches))
        rows = [row for batch in returned for row in batch]
        batch_count = len(batches)
    else:
        rows = evaluate_model_batch.remote(cases, model=model, revision=revision,
            prompts=tuple(prompt_names), max_turns=max_turns, max_calls=max_calls,
            max_new_tokens=max_new_tokens, allow_download=allow_model_download)
        batch_count = 1
    if len(rows) != episode_count:
        raise RuntimeError("Modal returned an unexpected episode count; no report was written")
    metadata = {
        "backend": "modal", "cloud_execution": True, "mode": mode, "status": "completed",
        "started_utc": started_at, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started, "modal_sdk_version": modal.__version__,
        "modal_app_id": app.app_id, "modal_dashboard_url": app.get_dashboard_url(),
        "available_cases": available, "evaluated_cases": len(cases), "episodes": len(rows),
        "requested_limit": limit, "variants": variants, "seeds": seeds,
        "workflows": _names(workflows), "conditions": _names(faults),
        "selection": "full_suite" if len(cases) == available else "prefix_smoke_test",
        "comparisons": list(POLICIES) if mode == "scripted" else prompt_names,
        "batch_count": batch_count, "max_tool_calls": max_calls,
        "model_calls": sum(row.get("model_call_count", 0) for row in rows),
        "policy_type": "rule_based_baselines" if mode == "scripted" else "real_local_cpu_model",
        "code_sha256": code_sha256,
    }
    finalize_run(rows, directory, metadata)
    print(f"Saved {len(rows)} returned cloud episodes to {directory}")
