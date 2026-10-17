"""Checks for reporting denominators and inspectable diagnosis evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from recovery_eval.core import build_cases, run_case
from recovery_eval.reporting import diagnosis_controls, diagnosis_rows, summarize, write_artifacts


class ReportingTests(unittest.TestCase):
    def test_summary_rates_use_episodes_within_each_group(self):
        case = build_cases(1, 1)[0]
        row = run_case(case, "bounded_retry")
        rows = []
        for fault, success, duplicates, missing, false_success, calls in (
            ("clean", True, 0, 0, False, 2),
            ("ack_lost", False, 2, 0, True, 4),
            ("ack_lost", False, 0, 1, False, 6),
        ):
            item = deepcopy(row)
            item.update(fault=fault, success=success, duplicates=duplicates,
                        missing=missing, false_success=false_success, tool_calls=calls)
            rows.append(item)
        second_policy = deepcopy(row)
        second_policy.update(policy="second_policy", seed=1)
        rows.append(second_policy)
        summary = summarize(rows)
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["task_payloads"], 1)
        self.assertEqual(summary["seeds"], 2)
        metrics = {(m["policy"], m["fault"]): m for m in summary["metrics"]}
        all_retry = metrics[("bounded_retry", "all")]
        self.assertEqual(all_retry["episodes"], 3)
        self.assertAlmostEqual(all_retry["success_rate"], 1 / 3)
        self.assertAlmostEqual(all_retry["duplicate_episode_rate"], 1 / 3)
        self.assertAlmostEqual(all_retry["missing_episode_rate"], 1 / 3)
        self.assertAlmostEqual(all_retry["false_success_rate"], 1 / 3)
        self.assertEqual(all_retry["mean_tool_calls"], 4)
        fault_retry = metrics[("bounded_retry", "ack_lost")]
        self.assertEqual(fault_retry["episodes"], 2)
        self.assertEqual(fault_retry["duplicate_episode_rate"], 0.5)
        self.assertEqual(metrics[("second_policy", "all")]["success_rate"], 1)

    def test_diagnosis_controls_are_real_executions_with_distinct_failures(self):
        controls = diagnosis_controls(build_cases(1, 2))
        by_variant = {row["case_id"].split(":")[-1]: row for row in controls}
        self.assertEqual(set(by_variant), {"empty", "record_only", "wrong_message", "wrong_link"})
        self.assertEqual(by_variant["empty"]["ledger"], [])
        self.assertEqual(by_variant["empty"]["missing"], 2)
        self.assertEqual(by_variant["record_only"]["missing"], 1)
        self.assertEqual(len(by_variant["record_only"]["ledger"]), 1)
        self.assertEqual(by_variant["wrong_message"]["wrong_payload"], 1)
        self.assertEqual(by_variant["wrong_link"]["wrong_payload"], 1)
        self.assertEqual(by_variant["wrong_link"]["missing"], 0)
        self.assertEqual(by_variant["wrong_message"]["final_state"]["notifications"][0]["message"], "Incorrect message")
        self.assertEqual(by_variant["wrong_link"]["final_state"]["notifications"][0]["record_id"], "nonexistent-record")
        for variant, row in by_variant.items():
            with self.subTest(variant=variant):
                actual_mutations = [e for e in row["trace"] if e["event"] == "tool_call"]
                self.assertEqual(len(actual_mutations), len(row["ledger"]))
                self.assertFalse(row["success"])
                self.assertEqual(row["evaluation_mode"], "known_answer_control")

    def test_diagnosis_labels_can_be_checked_using_only_given_evidence(self):
        case = build_cases(1, 1)[0]
        positive = run_case(case, "bounded_retry")
        controls = diagnosis_controls([case])
        duplicate_case = deepcopy(case)
        duplicate_case.update(fault="ack_lost", target_op="create_record")
        duplicate = run_case(duplicate_case, "bounded_retry")
        rows = diagnosis_rows([positive, *controls, duplicate])
        forbidden = {"fault", "target_op", "policy", "expected", "success", "duplicates", "missing", "ledger", "wrong_payload"}

        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        for row in rows:
            with self.subTest(case_id=row["id"]):
                evidence = {"request": row["request"], "trace": row["trace"]}
                self.assertFalse(keys(evidence) & forbidden)
                state = row["trace"][0]["response"]
                request = row["request"]
                records, notifications = state["records"], state["notifications"]
                # This is an independent known-goal check, using no raw episode labels.
                good_record = (len(records) == 1 and all(records[0].get(k) == request[k]
                                                      for k in ("task_id", "title", "body")))
                good_notification = (len(notifications) == 1 and good_record
                                     and notifications[0].get("record_id") == records[0]["record_id"]
                                     and all(notifications[0].get(k) == request[k]
                                             for k in ("task_id", "recipient", "message")))
                self.assertEqual(row["expected"]["success"], bool(good_record and good_notification))
                self.assertEqual(row["expected"]["missing"], int(not records) + int(not notifications))
                self.assertEqual(row["expected"]["duplicates"], max(0, len(records) - 1) + max(0, len(notifications) - 1))

    def test_report_escapes_identifiers_and_raw_trace_content(self):
        row = run_case(build_cases(1, 1)[0], "bounded_retry")
        dangerous = '<script>alert("injected")</script>'
        row["policy"] = dangerous
        row["case_id"] = dangerous
        row["trace"].append({"event": dangerous})
        with tempfile.TemporaryDirectory() as out:
            write_artifacts([row], out, metadata={"backend": dangerous})
            report = (Path(out) / "report.html").read_text(encoding="utf-8")
            self.assertNotIn(dangerous, report)
            self.assertIn("&lt;script&gt;", report)
            self.assertIn("&lt;/script&gt;", report)
            saved = json.loads((Path(out) / "episodes.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["case_id"], dangerous)

    def test_diagnosis_preserves_original_custom_business_payload(self):
        case = build_cases(1, 1)[0]
        case["task_id"] = "external-ticket-alpha"
        case["payload"].update(task_id=case["task_id"], title="Custom title", body="Custom issue",
                               recipient="custom@example.test", message="A custom message")
        result = run_case(case, "bounded_retry")
        row = diagnosis_rows([result])[0]
        for key, value in case["payload"].items():
            self.assertEqual(row["request"][key], value)
        self.assertTrue(row["expected"]["success"])

    def test_missing_original_payload_is_rejected_instead_of_guessed(self):
        row = run_case(build_cases(1, 1)[0], "bounded_retry")
        del row["task_payload"]
        with self.assertRaises(ValueError):
            diagnosis_rows([row])

    def test_live_model_report_is_not_labeled_as_scripted_baseline(self):
        row = run_case(build_cases(1, 1)[0], "bounded_retry")
        row.update(uses_language_model=True, evaluation_mode="live_model", policy="mock-model:basic")
        with tempfile.TemporaryDirectory() as out:
            write_artifacts([row], out, metadata={"experiment_kind": "live_model_recovery"})
            report = (Path(out) / "report.html").read_text(encoding="utf-8")
            manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("真实模型恢复评测", report)
            self.assertNotIn("不是 LLM 成绩", report)
            self.assertTrue(manifest["uses_language_model"])
            self.assertEqual(manifest["experiment_kind"], "live_model_recovery")

    def test_bundle_manifest_hashes_match_written_artifacts(self):
        row = run_case(build_cases(1, 1)[0], "bounded_retry")
        with tempfile.TemporaryDirectory() as out:
            summary = write_artifacts([row], out)
            directory = Path(out)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["episodes"], 1)
            self.assertFalse(manifest["uses_language_model"])
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), digest)

    def test_empty_results_are_rejected_before_creating_report(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(ValueError):
                write_artifacts([], out)
            self.assertEqual(list(Path(out).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
