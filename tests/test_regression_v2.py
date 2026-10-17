"""Check paired gates, cluster units, and honest failure accounting."""

import copy
import hashlib
import json
import unittest

from recovery_eval.regression import compare_runs


def rows(families=6, seeds=2, faults=("clean", "lost_response"), agent="before"):
    result = []
    for family in range(families):
        for seed in range(seeds):
            for fault in faults:
                config = {"variant_id": f"template-{family}", "params": {"item": family}, "seed": seed, "fault": fault}
                result.append({
                    "case_id": f"task-{family}-{seed}-{fault}", "workflow": "ticket", "fault": fault,
                    "seed": seed, "agent_id": agent, "agent_kind": "scripted", "success": True,
                    "duplicates": 0, "missing": 0, "false_success": False, "tool_calls": 3,
                    "condition_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    "case_config": config,
                })
    return result


class RegressionV2Tests(unittest.TestCase):
    def test_null_comparison_passes_with_zero_difference(self):
        source = rows()
        report = compare_runs(source, source)
        self.assertEqual(report["decision"], "pass")
        self.assertEqual(report["summary"]["success_delta_pp"], 0.0)
        self.assertEqual(report["summary"]["confidence_interval"]["lower_pp"], 0.0)
        self.assertEqual(report["summary"]["confidence_interval"]["independent_clusters"], 6)
        self.assertEqual(report["summary"]["paired_cases"], 24)

    def test_synthetic_regression_blocks_and_lists_case_ids(self):
        before = rows()
        after = copy.deepcopy(before)
        for row in after:
            row["success"] = False
        report = compare_runs(before, after)
        self.assertEqual(report["decision"], "block")
        self.assertEqual(report["summary"]["regressions"], len(after))
        self.assertEqual(report["summary"]["success_delta_pp"], -100.0)
        self.assertEqual(report["slices"]["by_fault"]["clean"]["regressions"], 12)
        self.assertEqual(report["summary"]["regression_case_ids"], sorted(row["case_id"] for row in after))

    def test_unpaired_case_sets_are_rejected(self):
        before = rows()
        with self.assertRaisesRegex(ValueError, "Case sets differ"):
            compare_runs(before, before[:-1])

    def test_duplicate_case_is_rejected(self):
        before = rows()
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            compare_runs(before + [before[0]], before)

    def test_missing_hash_or_changed_hash_are_rejected(self):
        before = rows()
        for value in (None, "different"):
            after = copy.deepcopy(before)
            after[0]["condition_hash"] = value
            with self.assertRaises(ValueError):
                compare_runs(before, after)

    def test_config_hash_alias_must_match(self):
        before = rows()
        after = copy.deepcopy(before)
        after[0]["config_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "disagree"):
            compare_runs(before, after)

    def test_case_config_mismatch_is_rejected_even_if_hash_was_not_updated(self):
        before = rows()
        after = copy.deepcopy(before)
        after[0]["case_config"]["params"]["item"] = "changed"
        with self.assertRaisesRegex(ValueError, "case_config"):
            compare_runs(before, after)

    def test_repeated_seeds_and_faults_do_not_inflate_cluster_count(self):
        source = rows(families=2, seeds=20)
        report = compare_runs(source, source)
        self.assertEqual(report["decision"], "inconclusive")
        interval = report["summary"]["confidence_interval"]
        self.assertEqual(interval["independent_clusters"], 2)
        self.assertEqual(interval["status"], "insufficient_evidence")
        self.assertIsNone(interval["lower_pp"])

    def test_missing_cluster_metadata_never_guesses_from_case_id(self):
        source = rows()
        for row in source:
            row.pop("case_config")
        report = compare_runs(source, source)
        self.assertEqual(report["decision"], "inconclusive")
        self.assertEqual(report["summary"]["confidence_interval"]["reason"], "missing_explicit_task_family")
        self.assertEqual(len(report["summary"]["confidence_interval"]["missing_cluster_case_ids"]), len(source))

    def test_safety_regression_blocks_even_when_mean_success_is_unchanged(self):
        before = rows(families=1)
        after = copy.deepcopy(before)
        after[0]["duplicates"] = 1
        report = compare_runs(before, after)
        self.assertEqual(report["decision"], "block")
        self.assertEqual(report["safety_gate"]["regressions"][0]["metrics"], ["duplicates"])

    def test_provider_errors_stay_in_denominator_and_prevent_pass(self):
        before = rows()
        after = copy.deepcopy(before)
        after[0]["run_status"] = "provider_error"
        report = compare_runs(before, after)
        self.assertEqual(report["decision"], "inconclusive")
        self.assertEqual(report["provider_errors"]["excluded_pairs"], 0)
        self.assertEqual(report["provider_errors"]["affected_pairs"], 1)
        self.assertEqual(report["summary"]["paired_cases"], len(before))

    def test_invalid_actions_are_not_labeled_as_provider_errors(self):
        source = rows()
        source[0]["run_status"] = "invalid_action"
        self.assertEqual(compare_runs(source, source)["provider_errors"]["affected_pairs"], 0)

    def test_agent_selection_prevents_implicit_agent_pooling(self):
        source = rows()
        additional = rows(agent="another")
        with self.assertRaisesRegex(ValueError, "exactly one explicit agent_id"):
            compare_runs(source + additional, source)
        selected = compare_runs(source + additional, source, before_agent="before")
        self.assertEqual(selected["summary"]["paired_cases"], len(source))

    def test_cross_kind_comparison_is_labeled_explicitly(self):
        before = rows()
        after = copy.deepcopy(before)
        for row in after:
            row["agent_kind"] = "model"
        report = compare_runs(before, after)
        self.assertEqual(report["comparison_kind"], "cross_agent_kind")
        self.assertEqual(report["after"]["agent_kind"], "model")

    def test_cost_and_call_deltas_do_not_treat_missing_cost_as_zero(self):
        before = rows()
        after = copy.deepcopy(before)
        before[0]["cost_usd"] = 0.01
        after[0]["cost_usd"] = 0.02
        after[0]["tool_calls"] = 5
        report = compare_runs(before, after)
        self.assertAlmostEqual(report["resources"]["cost_usd"]["delta"], 0.01)
        self.assertEqual(report["resources"]["cost_usd"]["paired_cases_with_measurement"], 1)
        self.assertEqual(report["summary"]["tool_call_delta"], 2)

    def test_cluster_bootstrap_is_deterministic(self):
        before = rows()
        after = copy.deepcopy(before)
        after[0]["success"] = False
        first = compare_runs(before, after)
        second = compare_runs(before, after)
        self.assertEqual(first, second)

    def test_threshold_and_confidence_are_validated(self):
        for invalid in (-1, 2, float("nan"), True):
            with self.assertRaises(ValueError):
                compare_runs(rows(), rows(), max_allowed_drop=invalid)
        for invalid in (0, 1, float("nan"), True):
            with self.assertRaises(ValueError):
                compare_runs(rows(), rows(), confidence=invalid)


if __name__ == "__main__":
    unittest.main()
