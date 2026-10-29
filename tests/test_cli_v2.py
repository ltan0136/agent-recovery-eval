"""Small end-to-end CLI runs and artifact evidence integrity checks.

No test downloads a model, invokes a paid endpoint, or launches a cloud job.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from recovery_eval import cli
from recovery_eval.artifacts import finalize_run, read_jsonl, summarize
from recovery_eval.engine import run_scripted
from recovery_eval.workflows import build_matrix


class CliV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "recovery_eval", *map(str, args)],
                              cwd=Path(__file__).resolve().parents[1], capture_output=True,
                              text=True, timeout=20, check=False)

    def invoke(self, args):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            cli.main(list(map(str, args)))
        return output.getvalue()

    def scripted_row(self):
        return run_scripted(build_matrix("ticket_dispatch", 1, 1, "clean")[0])

    def test_real_small_matrix_run_replay_and_manifest_hashes(self):
        matrix = self.root / "cases.jsonl"
        exported = self.run_cli("matrix", "--workflows", "ticket_dispatch", "--variants", 2,
                                "--seeds", 1, "--faults", "clean,lost_ack", "--out", matrix)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        self.assertEqual(len(read_jsonl(matrix)), 4)
        output = self.root / "run"
        completed = self.run_cli("run", "--workflows", "ticket_dispatch", "--variants", 2,
                                 "--seeds", 1, "--faults", "clean,lost_ack", "--policies",
                                 "idempotent_retry,bounded_retry", "--out", output)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        episodes = read_jsonl(output / "episodes.jsonl")
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["episodes"], 8)
        self.assertEqual(summary["case_conditions"], 4)
        self.assertEqual(summary["task_families"], 2)
        self.assertEqual(summary["agent_kinds"], ["scripted"])
        self.assertEqual(read_jsonl(output / "progress.jsonl"), episodes)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertFalse(manifest["source_changed_during_run"])
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
        saved = episodes[0]
        replayed = self.run_cli("replay", "--episodes", output / "episodes.jsonl", "--case-id",
                                saved["case_id"], "--policy", saved["agent_id"])
        self.assertEqual(replayed.returncode, 0, replayed.stderr)
        self.assertEqual(json.loads(replayed.stdout), saved)

    def test_compare_block_persists_evidence_and_sets_ci_exit_code(self):
        cases = build_matrix("ticket_dispatch", 1, 1, "lost_ack")
        rows = [run_scripted(c, policy) for c in cases for policy in ("idempotent_retry", "bounded_retry")]
        episodes = self.root / "episodes.jsonl"
        episodes.write_text("".join(json.dumps(row) + "\n" for row in rows))
        destination = self.root / "comparison.json"
        compared = self.run_cli("compare", "--before", episodes, "--after", episodes,
                                "--before-agent", "idempotent_retry", "--after-agent", "bounded_retry",
                                "--out", destination, "--fail-on-block")
        self.assertEqual(compared.returncode, 2, compared.stderr)
        report = json.loads(destination.read_text())
        self.assertEqual(report["decision"], "block")
        self.assertEqual(report["summary"]["paired_cases"], 1)
        self.assertEqual(report["provider_errors"]["excluded_pairs"], 0)
        self.assertTrue(report["safety_gate"]["regressions"])

    def test_existing_run_directory_is_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        orphaned = output / "episodes.jsonl"
        orphaned.write_text("preserve even an incomplete prior run\n")
        with self.assertRaises(SystemExit) as caught:
            self.invoke(["run", "--workflows", "ticket_dispatch", "--variants", 1, "--seeds", 1,
                         "--faults", "clean", "--out", output])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(orphaned.read_text(), "preserve even an incomplete prior run\n")
        self.assertEqual(list(output.iterdir()), [orphaned])

    def test_existing_matrix_comparison_and_diagnosis_outputs_are_protected(self):
        destination = self.root / "existing.jsonl"
        destination.write_text("original evidence\n")
        commands = [
            ["matrix", "--variants", 1, "--seeds", 1, "--faults", "clean", "--out", destination],
            ["compare", "--before", "absent", "--after", "absent", "--before-agent", "a",
             "--after-agent", "b", "--out", destination],
            ["export-diagnosis", "--episodes", "absent", "--out", destination],
        ]
        for arguments in commands:
            with self.subTest(command=arguments[0]):
                with self.assertRaises(SystemExit):
                    self.invoke(arguments)
                self.assertEqual(destination.read_text(), "original evidence\n")

    def test_interrupt_preserves_progress_and_start_provenance(self):
        output = self.root / "interrupted"
        original = run_scripted
        completed = 0

        def interrupt_after_one(case, policy, **kwargs):
            nonlocal completed
            if completed:
                raise KeyboardInterrupt()
            completed += 1
            return original(case, policy, **kwargs)

        with patch("recovery_eval.cli.run_scripted", side_effect=interrupt_after_one):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(["run", "--workflows", "ticket_dispatch", "--variants", 1,
                             "--seeds", 1, "--faults", "clean,lost_ack", "--policies",
                             "idempotent_retry", "--max-tool-calls", 17, "--out", output])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["completed_episodes"], 1)
        self.assertEqual(manifest["max_tool_calls"], 17)
        self.assertIn("engine.py", manifest["code_sha256"])
        self.assertEqual(manifest["policies"], ["idempotent_retry"])
        self.assertEqual(len(read_jsonl(output / "progress.jsonl")), 1)
        self.assertFalse((output / "episodes.jsonl").exists())
        self.assertFalse((output / "summary.json").exists())

    def test_interrupted_model_configuration_is_retained_without_model_execution(self):
        output = self.root / "model-interrupted"
        adapter = Mock()
        adapter.describe.return_value = {"backend": "test_fixture", "model_id": "fixture-only"}
        with patch("recovery_eval.agents.OpenAICompatibleAdapter", return_value=adapter), \
                patch("recovery_eval.agents.run_agent", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(["model", "--backend", "api", "--model", "fixture-only",
                             "--base-url", "https://example.invalid/v1", "--workflows", "ticket_dispatch",
                             "--faults", "clean", "--max-turns", 5, "--max-new-tokens", 99,
                             "--max-tool-calls", 7, "--out", output])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["requested_model"], "fixture-only")
        self.assertEqual(manifest["max_turns"], 5)
        self.assertEqual(manifest["max_new_tokens"], 99)
        self.assertEqual(manifest["max_tool_calls"], 7)
        self.assertEqual(manifest["model_backend"]["backend"], "test_fixture")

    def test_replay_selects_matching_policy_configuration_and_rejects_tampering(self):
        case = build_matrix("ticket_dispatch", 1, 1, "clean")[0]
        first = run_scripted(case, "bounded_retry", max_calls=1)
        intended = run_scripted(case, "idempotent_retry", max_calls=24)
        episodes = self.root / "replay.jsonl"
        episodes.write_text(json.dumps(first) + "\n" + json.dumps(intended) + "\n")
        output = self.invoke(["replay", "--episodes", episodes, "--case-id", case["case_id"],
                              "--policy", "idempotent_retry"])
        self.assertEqual(json.loads(output), intended)
        intended["condition_hash"] = "tampered"
        episodes.write_text(json.dumps(intended) + "\n")
        with self.assertRaises(SystemExit):
            self.invoke(["replay", "--episodes", episodes, "--case-id", case["case_id"],
                         "--policy", "idempotent_retry"])

    def test_report_escapes_untrusted_text_and_keeps_agent_kinds_separate(self):
        scripted = self.scripted_row()
        # Synthetic fixtures test the reporting contract; no real model is run.
        model = deepcopy(scripted)
        attack = "</script><script>alert('untrusted')</script>&"
        model.update({"agent_id": attack, "agent_kind": "local_model", "success": False,
                      "run_status": "invalid_action", "model_calls": [
                          {"status": "invalid_action", "raw_generation": attack,
                           "usage": {"completion_tokens": 9}, "metadata": {"generation_truncated": True}}
                      ]})
        output = self.root / "safe-report"
        summary = finalize_run([scripted, model], output)
        self.assertEqual(summary["agent_kinds"], ["local_model", "scripted"])
        totals = {row["agent_id"]: row for row in summary["metrics"]
                  if row["workflow"] == row["fault"] == "all"}
        self.assertEqual(totals[scripted["agent_id"]]["success_rate"], 1.0)
        self.assertEqual(totals[attack]["success_rate"], 0.0)
        self.assertEqual(totals[attack]["invalid_model_responses"], 1)
        self.assertEqual(totals[attack]["truncated_model_responses"], 1)
        self.assertEqual(totals[attack]["provider_error_episodes"], 0)
        page = (output / "report.html").read_text()
        self.assertNotIn(attack, page)
        self.assertIn("&lt;/script&gt;", page)
        data = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL)
        self.assertIsNotNone(data)
        payload = json.loads(data.group(1))
        self.assertEqual(payload["traces"][1]["model_calls"][0]["raw_generation"], attack)

    def test_summary_rejects_duplicate_rows_and_mixed_agent_identity(self):
        row = self.scripted_row()
        with self.assertRaisesRegex(ValueError, "Duplicate case"):
            summarize([row, deepcopy(row)])
        changed = deepcopy(row)
        changed.update({"case_id": "different-case", "agent_kind": "local_model"})
        with self.assertRaisesRegex(ValueError, "different agent kinds"):
            summarize([row, changed])

    def test_invalid_run_arguments_leave_no_output_directory(self):
        output = self.root / "typo"
        for args in (["run", "--policies", "idempotent_retry,not_a_policy"],
                     ["model", "--prompts", "basic,unknown"],
                     ["model", "--backend", "api", "--model", "m"]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.invoke([*args, "--out", output])
            self.assertFalse(output.exists())

    def test_seed_consistency_requires_every_seed_of_a_family_to_pass(self):
        cases = build_matrix("ticket_dispatch", 2, 2, "clean")
        rows = [run_scripted(case) for case in cases]
        self.assertTrue(all(row["success"] for row in rows))
        rows[0]["success"] = False  # variant 1, seed 0 fails; seed 1 of that variant still passes
        consistency = {c["fault"]: c for c in summarize(rows)["seed_consistency"]}
        for fault in ("all", "clean"):
            self.assertEqual(consistency[fault]["families"], 2)
            self.assertEqual(consistency[fault]["k"], 2)
            self.assertEqual(consistency[fault]["all_seeds_successes"], 1)
            self.assertEqual(consistency[fault]["all_seeds_success_rate"], 0.5)
            self.assertEqual(consistency[fault]["per_episode_success_rate"], 0.75)

    def test_report_shows_pass_k_only_with_repeated_seeds(self):
        single = finalize_run([self.scripted_row()], self.root / "one-seed")
        self.assertEqual(single["seed_consistency"][0]["k"], 1)
        self.assertNotIn("(k=", (self.root / "one-seed" / "report.html").read_text())
        rows = [run_scripted(case) for case in build_matrix("ticket_dispatch", 1, 3, "clean")]
        finalize_run(rows, self.root / "three-seeds")
        self.assertIn("100.0% (k=3)", (self.root / "three-seeds" / "report.html").read_text())

    def test_artifact_verifier_accepts_summaries_written_before_pass_k(self):
        from scripts.verify_artifacts import verify
        rows = [run_scripted(case) for case in build_matrix("ticket_dispatch", 1, 2, "clean")]
        output = self.root / "legacy"
        finalize_run(rows, output, {"command": "test"})
        self.assertEqual(verify(output)["episodes"], 2)
        summary_path = output / "summary.json"
        legacy = json.loads(summary_path.read_text())
        del legacy["seed_consistency"]
        summary_path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2) + "\n")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files_sha256"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(verify(output)["episodes"], 2)
        legacy["episodes"] = 3
        summary_path.write_text(json.dumps(legacy))
        manifest["files_sha256"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Summary differs"):
            verify(output)

    def test_finalization_preserves_start_hashes_and_marks_source_changes(self):
        output = self.root / "provenance"
        with patch("recovery_eval.artifacts.source_hashes", return_value={"engine.py": "end"}):
            finalize_run([self.scripted_row()], output,
                         {"code_sha256": {"engine.py": "start"}, "created_utc": "start-time"})
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["code_sha256"], {"engine.py": "start"})
        self.assertEqual(manifest["code_sha256_end"], {"engine.py": "end"})
        self.assertTrue(manifest["source_changed_during_run"])
        self.assertEqual(manifest["changed_source_files"], ["engine.py"])
        self.assertEqual(manifest["created_utc"], "start-time")
        before = (output / "episodes.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            finalize_run([self.scripted_row()], output)
        self.assertEqual((output / "episodes.jsonl").read_bytes(), before)

    def test_diagnosis_export_preserves_settled_state_and_ledger_only(self):
        row = self.scripted_row()
        episodes = self.root / "episodes.jsonl"
        episodes.write_text(json.dumps(row) + "\n")
        output = self.root / "diagnosis.jsonl"
        self.invoke(["export-diagnosis", "--episodes", episodes, "--out", output])
        exported = read_jsonl(output)[0]
        self.assertEqual(exported["trace"][0]["result"], row["final_state"])
        self.assertEqual(exported["trace"][1]["result"], row["ledger"])
        self.assertEqual(set(exported["expected"]), {"duplicates", "missing", "success"})
        self.assertNotIn("fault", exported["request"])
        self.assertNotIn("target_step", exported["request"])


if __name__ == "__main__":
    unittest.main()
