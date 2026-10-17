"""Continuation evidence tests using a fake adapter and the real simulator.

The fixture emits known invalid text; these are not language-model experiments.
"""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from recovery_eval.agents import ModelResponse, run_agent
from recovery_eval.artifacts import finalize_run, read_jsonl, source_hashes
from recovery_eval.workflows import build_matrix, workflow_specs
from scripts import resume_model_run as resume


class FixtureAdapter:
    """Deterministic fake adapter. No weights, endpoint, or inference are used."""

    kind = "local_model"

    def __init__(self, model_id, *, revision, **kwargs):
        self.model_id = model_id
        self.revision = revision

    def complete(self, messages):
        return ModelResponse(text="fixture deliberately invalid JSON", generated_token_ids=[11, 12],
                             usage={"completion_tokens": 2}, finish_reason="eos",
                             metadata={"fixture_only": True, "generation_truncated": False})

    def describe(self):
        return {"backend": "fixture_no_model_inference", "model_id": self.model_id,
                "requested_revision": self.revision, "resolved_revision_sha": self.revision,
                "device": "cpu", "dtype": "float32", "do_sample": False,
                "threads": 4, "max_new_tokens": 160, "fixture_only": True}


@unittest.skipIf(sys.platform == "win32", "Continuation currently requires Unix fcntl")
class ResumeV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "interrupted-source"
        self.source.mkdir()
        self.output = self.root / "continuation"
        self.record = self.root / "interruption.json"
        self.record.write_text(json.dumps({"fixture_only": True, "inflight_attempt": "unknown"}))
        self.model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        self.jobs = [(case, prompt) for case in build_matrix(None, 1, 1, ["clean", "lost_ack"])
                     for prompt in ["basic", "guarded"]]

    def tearDown(self):
        self.temporary.cleanup()

    def make_source(self, prefix=7):
        adapter = FixtureAdapter(self.model, revision=self.revision)
        rows = [run_agent(case, adapter, prompt_variant=prompt, max_turns=8,
                          max_tool_calls=12, format_retries=1) for case, prompt in self.jobs[:prefix]]
        self.source.joinpath("progress.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        metadata = {"command": "model", "status": "interrupted", "requested_backend": "local",
                    "requested_workflows": list(workflow_specs()), "variants": 1, "seeds": 1,
                    "conditions": ["clean", "lost_ack"], "prompt_variants": ["basic", "guarded"],
                    "max_turns": 8, "max_tool_calls": 12, "max_new_tokens": 160, "format_retries": 1,
                    "requested_model": self.model, "requested_revision": self.revision,
                    "model_backend": adapter.describe(), "code_sha256": source_hashes(),
                    "created_utc": datetime.now(timezone.utc).isoformat(), "cloud_execution": False,
                    "fixture_only": True}
        self.source.joinpath("manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        return rows

    def continue_one(self):
        arguments = ["resume_model_run.py", "--source", str(self.source), "--out", str(self.output),
                     "--cache-dir", str(self.root / "unused-cache"), "--interruption-record", str(self.record), "--one"]
        output = io.StringIO()
        with patch.object(sys, "argv", arguments), patch.object(resume, "LocalTransformersAdapter", FixtureAdapter), \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            resume.main()
        return output.getvalue()

    def test_original_failures_and_each_one_episode_checkpoint_are_preserved(self):
        original = self.make_source(7)
        original_bytes = self.source.joinpath("progress.jsonl").read_bytes()
        self.assertTrue(all(not row["success"] for row in original))
        self.continue_one()
        eight = read_jsonl(self.output / "progress.jsonl")
        self.assertEqual(eight[:7], original)
        self.assertEqual(len(eight), 8)
        self.assertFalse((self.output / "episodes.jsonl").exists())
        self.continue_one()
        nine = read_jsonl(self.output / "progress.jsonl")
        self.assertEqual(nine[:8], eight)
        self.assertEqual(len(nine), 9)
        self.assertEqual(len({(r["case_id"], r["agent_id"]) for r in nine}), 9)
        self.assertEqual(self.source.joinpath("progress.jsonl").read_bytes(), original_bytes)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["preserved_original_episodes"], 7)
        self.assertEqual([s["new_episodes"] for s in manifest["continuation_sessions"]], [1, 1])
        self.assertIn("unknown", manifest["unobserved_inflight_attempt"])
        self.assertIn("lower bound", manifest["timing_note"])

    def test_changed_source_and_concurrent_worker_are_rejected(self):
        self.make_source()
        path = self.source / "manifest.json"
        original = path.read_bytes()
        metadata = json.loads(original)
        metadata["code_sha256"]["engine.py"] = "source was changed"
        path.write_text(json.dumps(metadata))
        with self.assertRaises(SystemExit):
            self.continue_one()
        self.assertFalse(self.output.exists())
        path.write_bytes(original)
        self.output.mkdir()
        import fcntl
        with (self.output / "continuation.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(SystemExit):
                self.continue_one()
        self.assertFalse((self.output / "progress.jsonl").exists())

    def test_unsupported_budgets_and_incomplete_workflow_protocol_are_rejected(self):
        self.make_source(1)
        path = self.source / "manifest.json"
        original = json.loads(path.read_text())
        for field, value in (("max_turns", 9), ("requested_workflows", ["ticket_dispatch"])):
            with self.subTest(field=field):
                metadata = dict(original)
                metadata[field] = value
                path.write_text(json.dumps(metadata))
                with self.assertRaises(SystemExit):
                    self.continue_one()

    def test_corrupt_frozen_episode_configuration_is_rejected(self):
        rows = self.make_source(1)
        rows[0]["execution_config"]["max_calls"] = 11
        self.source.joinpath("progress.jsonl").write_text(json.dumps(rows[0]) + "\n")
        with self.assertRaises(SystemExit):
            self.continue_one()

    def test_finalize_occurs_only_at_sixteen_and_completed_run_is_not_duplicated(self):
        original = self.make_source(14)
        with patch.object(resume, "finalize_run", wraps=finalize_run) as finalize:
            self.continue_one()
            self.assertEqual(len(read_jsonl(self.output / "progress.jsonl")), 15)
            finalize.assert_not_called()
            self.continue_one()
            finalize.assert_called_once()
            episodes = read_jsonl(self.output / "episodes.jsonl")
            self.assertEqual(len(episodes), 16)
            self.assertEqual(episodes[:14], original)
            self.assertEqual(episodes, read_jsonl(self.output / "progress.jsonl"))
            manifest = json.loads((self.output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["completed_episodes"], 16)
            completed = self.continue_one()
            self.assertIn("already_completed", completed)
            finalize.assert_called_once()
            self.assertEqual(read_jsonl(self.output / "episodes.jsonl"), episodes)


if __name__ == "__main__":
    unittest.main()
