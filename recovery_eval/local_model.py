"""Evaluate a local causal LM on settled trace diagnosis, not recovery behavior."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_PROMPT = (
    "You diagnose a completed tool execution. The request requires exactly one "
    "record and exactly one notification linked to that record. "
    "Use the settled final state in the trace, not the agent's success claim. "
    "duplicates is the number of extra committed records and notifications "
    "beyond one of each. missing is the number of required operation types "
    "with zero committed effects. success is true only when exactly one record "
    "and one correctly linked notification exist and the request is satisfied. "
    "Return only one JSON object with exactly these three keys: "
    '"duplicates" (integer), "missing" (integer), "success" (boolean). '
    "Do not output explanations or Markdown."
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_answer(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"duplicates", "missing", "success"}
        and type(value["duplicates"]) is int
        and value["duplicates"] >= 0
        and type(value["missing"]) is int
        and value["missing"] >= 0
        and type(value["success"]) is bool
    )


def score_response(raw: str, expected: dict[str, Any]) -> dict[str, Any]:
    """Parse the entire raw completion without extracting or repairing JSON."""
    try:
        prediction = json.loads(
            raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, TypeError) as error:
        return {
            "json_valid": False,
            "strict_json_valid": False,
            "exact_match": False,
            "prediction": None,
            "parse_error": str(error),
        }
    valid = _is_answer(prediction)
    return {
        "json_valid": True,
        "strict_json_valid": valid,
        "exact_match": valid and prediction == expected,
        "prediction": prediction,
        "parse_error": None if valid else "Output does not match the exact answer schema.",
    }


def build_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    """Expose only public request and settled trace; never expose expected labels."""
    public_input = {"request": case["request"], "trace": case["trace"]}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(public_input, ensure_ascii=False)},
    ]


def load_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    cases = []
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            required = {"id", "request", "trace", "expected"}
            if not required <= set(case) or not _is_answer(case["expected"]):
                raise ValueError(f"Invalid case schema at line {line_number}")
            if case["id"] in ids:
                raise ValueError(f"Duplicate case ID: {case['id']}")
            ids.add(case["id"])
            cases.append(case)
            if len(cases) == limit:
                break
    if not cases:
        raise ValueError("The input contains no cases.")
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_description() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.threads < 1 or args.max_new_tokens < 1:
        parser.error("limit, threads and max-new-tokens must be positive.")
    cases = load_cases(args.data, args.limit)

    # Optional heavyweight dependencies are imported only for model execution.
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    loading_options = {
        "revision": args.revision,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "local_files_only": args.local_files_only,
        "token": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **loading_options)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, **loading_options
    ).to("cpu").eval()
    load_seconds = time.perf_counter() - started
    revision_sha = getattr(model.config, "_commit_hash", None)
    if not revision_sha:
        # New Transformers versions may omit _commit_hash from the model config.
        cached_config = try_to_load_from_cache(
            args.model, "config.json", revision=args.revision,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        if isinstance(cached_config, str):
            revision_sha = Path(cached_config).parent.name
    if not revision_sha or len(revision_sha) != 40:
        raise ValueError("Cannot resolve the model revision SHA from the local cache.")
    results = []
    for case in cases:
        prompt = tokenizer.apply_chat_template(
            build_messages(case), tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(encoded["input_ids"].shape[-1])
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and input_tokens + args.max_new_tokens > context_limit:
            raise ValueError(
                f"Case {case['id']} exceeds context limit; no truncation was applied."
            )
        generation_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                use_cache=True,
            )
        seconds = time.perf_counter() - generation_start
        generated_ids = output_ids[0, input_tokens:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = int(generated_ids.shape[-1])
        eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        ended_with_eos = bool(generated_tokens and int(generated_ids[-1]) in eos_ids)
        limit_reached = generated_tokens == args.max_new_tokens
        result = {
            "id": case["id"],
            "prompt": prompt,
            "raw_generation": raw,
            "expected": case["expected"],
            **score_response(raw, case["expected"]),
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "generated_token_ids": generated_ids.tolist(),
            "generation_seconds": seconds,
            "generation_tokens_per_second": generated_tokens / seconds if seconds else None,
            "generation_limit_reached": limit_reached,
            "finish_reason": "eos" if ended_with_eos else ("max_new_tokens" if limit_reached else "other"),
            "generation_truncated": limit_reached and not ended_with_eos,
            "finish_reason_source": "Generated token IDs and configured EOS token IDs.",
        }
        results.append(result)
        print(json.dumps({"id": case["id"], "strict_json_valid": result["strict_json_valid"],
                          "exact_match": result["exact_match"], "seconds": round(seconds, 3)}),
              flush=True)

    report = {
        "evaluation_type": "local_language_model_settled_trace_diagnosis",
        "scope": "Synthetic trace diagnosis only; not language-model agent recovery performance.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": args.model, "requested_revision": args.revision,
            "resolved_revision_sha": revision_sha, "device": "cpu", "dtype": "float32",
            "do_sample": False, "max_new_tokens": args.max_new_tokens,
            "batch_size": 1, "seed": 0, "threads": args.threads,
        },
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu": _cpu_description(), "logical_cpu_count": os.cpu_count(),
            "versions": {name: _package_version(name) for name in
                         ("torch", "transformers", "huggingface_hub", "tokenizers", "safetensors")},
        },
        "provenance": {
            "data_file": str(args.data), "data_sha256": _sha256(args.data),
            "evaluator_sha256": _sha256(Path(__file__)),
            "expected_labels_in_prompt": False, "output_repair_applied": False,
            "input_truncation_applied": False,
        },
        "summary": {
            "cases": len(results), "model_load_seconds": load_seconds,
            "strict_json_valid_count": sum(row["strict_json_valid"] for row in results),
            "exact_match_count": sum(row["exact_match"] for row in results),
            "strict_json_valid_rate": sum(row["strict_json_valid"] for row in results) / len(results),
            "exact_match_rate": sum(row["exact_match"] for row in results) / len(results),
            "input_tokens": sum(row["input_tokens"] for row in results),
            "generated_tokens": sum(row["generated_tokens"] for row in results),
            "generation_seconds": sum(row["generation_seconds"] for row in results),
            "sampling_note": "First N input cases; no random sampling or population-generalization claim.",
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
