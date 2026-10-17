"""Reproduce the release matrix and optional frozen-model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    ("smol", "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
    ("qwen", "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/reproduction")
    parser.add_argument("--models", action="store_true", help="Also run the two CPU models; requires optional dependencies")
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        parser.error("Choose a new output directory")
    out.mkdir(parents=True)

    def cli(*parts, expected=0):
        completed = subprocess.run([sys.executable, "-m", "recovery_eval", *map(str, parts)], cwd=ROOT)
        if completed.returncode != expected:
            raise SystemExit(f"Unexpected exit {completed.returncode}, expected {expected}: {parts}")

    cli("run", "--variants", 12, "--seeds", 4, "--out", out / "scripted")
    episodes = out / "scripted" / "episodes.jsonl"
    common = ("compare", "--before", episodes, "--after", episodes, "--before-agent", "idempotent_retry")
    cli(*common, "--after-agent", "idempotent_retry", "--out", out / "unchanged.json", "--fail-on-block")
    cli(*common, "--after-agent", "bounded_retry", "--out", out / "intentional-regression.json", "--fail-on-block", expected=2)
    for name, expected_decision in (("unchanged", "pass"), ("intentional-regression", "block")):
        observed = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        if observed["decision"] != expected_decision:
            raise SystemExit(f"Unexpected decision for {name}: {observed['decision']}")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_artifacts.py"),
                    str(out / "scripted")], cwd=ROOT, check=True)
    if args.models:
        for name, model, revision in MODELS:
            options = ["--cache-dir", args.cache_dir] if args.cache_dir else []
            if args.allow_download:
                options.append("--allow-download")
            cli("model", "--model", model, "--revision", revision,
                "--variants", 1, "--seeds", 1, "--faults", "clean,lost_ack", "--prompts", "basic,guarded",
                "--max-new-tokens", 160, "--max-turns", 8, "--max-tool-calls", 12,
                "--format-retries", 1, "--threads", 4, "--out", out / f"model-{name}", *options)
            subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_artifacts.py"),
                            str(out / f"model-{name}")], cwd=ROOT, check=True)
    print("Results:", out)


if __name__ == "__main__":
    main()
