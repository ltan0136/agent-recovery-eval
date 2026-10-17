"""One-command local reproduction with no cloud credentials or dependencies."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=root, check=True)
subprocess.run([sys.executable, "-m", "recovery_eval", "demo", "--out", "results/local"], cwd=root, check=True)
subprocess.run([sys.executable, "-m", "recovery_eval", "compare",
                "--before", "results/local/episodes.jsonl", "--after", "results/local/episodes.jsonl",
                "--before-policy", "idempotent_retry", "--after-policy", "verify_then_retry",
                "--out", "results/regression"], cwd=root, check=True)
print("Open:", root / "results/local/report.html")
print("Regression report:", root / "results/regression/report.html")
