"""Run a credential-free v2 demo, preserving earlier runs."""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = root / "runs" / f"demo-{stamp}"
subprocess.run([sys.executable, "-m", "recovery_eval", "run", "--variants", "2", "--seeds", "1",
                "--out", str(output)], cwd=root, check=True)
subprocess.run([sys.executable, "-m", "recovery_eval", "compare",
                "--before", str(output / "episodes.jsonl"), "--after", str(output / "episodes.jsonl"),
                "--before-agent", "idempotent_retry", "--after-agent", "bounded_retry",
                "--out", str(output / "regression.json")], cwd=root, check=True)
print("Open:", output / "report.html")
print("Expected regression block:", output / "regression.json")
