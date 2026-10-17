"""Validate actual Verifiers v1 Taskset loading and rewards without inference."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

try:
    from recovery_diagnosis import DiagnosisConfig, DiagnosisTaskset, load_environment
    from verifiers.v1 import Trace
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.runtimes.subprocess import SubprocessConfig
    from verifiers.v1.trace import AgentInfo, TraceTask
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


@unittest.skipUnless(SDK_AVAILABLE, "Install verifiers==0.4.0 for native Taskset checks")
class NativeTasksetTests(unittest.TestCase):
    def test_fixture_loads_all_workflows_without_reference_leakage(self):
        tasks = load_environment().load()
        self.assertEqual(len(tasks), 40)
        self.assertEqual(len({task.data.id for task in tasks}), 40)
        self.assertIsInstance(load_environment(), DiagnosisTaskset)
        workflows = set()
        for task in tasks:
            prompt = task.data.prompt
            evidence = json.loads(prompt.split("BEGIN_OBSERVED_EVIDENCE\n", 1)[1].split("\nEND_OBSERVED_EVIDENCE", 1)[0])
            workflows.add(evidence["request"]["workflow_id"])
            self.assertEqual(set(evidence), {"request", "trace"})
            self.assertEqual([item["tool"] for item in evidence["trace"]],
                             ["inspect_settled_state", "inspect_settled_ledger"])
            for private in ('"expected":', '"fault":', '"target_step":', '"case_config":'):
                self.assertNotIn(private, prompt)
        self.assertEqual(len(workflows), 4)

    def test_native_reward_accepts_reference_and_rejects_wrong_reply(self):
        for task in load_environment().load():
            with self.subTest(case=task.data.id):
                trace = Trace(task=TraceTask(type="recovery_diagnosis", data=task.data),
                    agent=AgentInfo(config=AgentConfig(runtime=SubprocessConfig())),
                    root_reply=json.dumps(task.data.expected))
                self.assertEqual(asyncio.run(task.diagnosis_exact(trace)), 1.0)
                trace.root_reply = "not a JSON answer"
                self.assertEqual(asyncio.run(task.diagnosis_exact(trace)), 0.0)

    def test_wrong_payload_controls_do_not_equate_zero_counts_with_success(self):
        tasks = [task for task in load_environment().load() if "known_wrong_payload" in task.data.id]
        self.assertEqual(len(tasks), 4)
        for task in tasks:
            self.assertEqual(task.data.expected, {"duplicates": 0, "missing": 0, "success": False})
            self.assertIn("necessary but not sufficient", task.data.prompt)

    def test_loader_rejects_duplicate_ids_and_private_evidence(self):
        source = Path(__file__).resolve().parents[1] / "data" / "diagnosis_cases.jsonl"
        row = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                DiagnosisTaskset(DiagnosisConfig(data_path=str(path))).load()
            row["trace"][0]["result"]["fault"] = "lost_ack"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reference label"):
                DiagnosisTaskset(DiagnosisConfig(data_path=str(path))).load()

    def test_loader_rejects_incompatible_old_observation_schema(self):
        source = Path(__file__).resolve().parents[1] / "data" / "diagnosis_cases.jsonl"
        row = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
        row["trace"][0]["response"] = row["trace"][0].pop("result")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tool/result schema"):
                DiagnosisTaskset(DiagnosisConfig(data_path=str(path))).load()


if __name__ == "__main__":
    unittest.main()
