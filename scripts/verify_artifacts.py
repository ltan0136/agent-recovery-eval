"""Verify a run's saved file checksums and summary against its raw episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_eval.artifacts import read_jsonl, summarize
from recovery_eval.workflows import config_hash


def verify(directory: Path) -> dict:
    """Verify the complete saved denominator and its frozen input evidence."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Run is not complete")
    if manifest.get("source_changed_during_run"):
        raise ValueError("Run source changed during execution")
    if manifest.get("code_sha256_end") is not None and manifest.get("code_sha256") != manifest["code_sha256_end"]:
        raise ValueError("Run start and end source hashes differ")
    hashes = manifest.get("files_sha256", {})
    required = {"episodes.jsonl", "summary.json", "metrics.csv", "example_traces.json", "report.html"}
    if manifest.get("command") in ("run", "model"):
        required.add("progress.jsonl")
    if not required.issubset(hashes):
        raise ValueError(f"Manifest omits required artifacts: {sorted(required - set(hashes))}")
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Manifest filename must be a basename")
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    rows = read_jsonl(directory / "episodes.jsonl")
    saved = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if saved != summarize(rows):
        raise ValueError("Summary differs from the raw episodes")
    if (directory / "progress.jsonl").exists() and read_jsonl(directory / "progress.jsonl") != rows:
        raise ValueError("Final episodes differ from the complete progress evidence")
    for row in rows:
        frozen = config_hash(row["execution_config"])
        if row.get("condition_hash") != frozen or row.get("config_hash") != frozen:
            raise ValueError(f"Execution condition hash mismatch: {row['case_id']}")
        if row.get("case_hash") != config_hash(row["case_config"]):
            raise ValueError(f"Case configuration hash mismatch: {row['case_id']}")
        frozen_case = {key: value for key, value in row["case_config"].items() if key != "config_hash"}
        if frozen_case != row["execution_config"].get("case"):
            raise ValueError(f"Case and execution configuration disagree: {row['case_id']}")
        for field in ("case_id", "workflow", "fault", "seed", "variant_id"):
            if row[field] != row["case_config"][field]:
                raise ValueError(f"Result metadata differs from the frozen case: {row['case_id']}/{field}")
    return {"verified_files": len(hashes), "episodes": saved["episodes"],
            "condition_hashes_recomputed": len(rows), "source_changed_during_run": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        checked = verify(args.directory)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(checked))


if __name__ == "__main__":
    main()
