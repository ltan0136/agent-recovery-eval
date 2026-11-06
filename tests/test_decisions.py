"""Retry-permission audit and conditional recovery (CRSR) from saved episodes."""

from pathlib import Path
import tempfile
import unittest

from recovery_eval.artifacts import finalize_run, summarize
from recovery_eval.decisions import conditional_recovery, retry_audit, retry_permission
from recovery_eval.engine import Environment, run_scripted
from recovery_eval.workflows import build_matrix


def case_for(fault, workflow="ticket_dispatch"):
    return build_matrix(workflows=workflow, variants=1, seeds=1, faults=fault)[0]


def run_actions(case, keys, *, idempotency=None):
    """Write every planned step; repeat the target step once with the given keys."""
    environment = Environment(case, idempotency=idempotency)
    for step in environment.public_task["steps"]:
        attempts = keys if step["name"] == case["target_step"] else [keys[-1]]
        for key in attempts:
            arguments = dict(step["arguments"])
            if key is not None:
                arguments["operation_id"] = key
            if environment.invoke(step["tool"], arguments)["ok"]:
                break
    return environment.finish(True, "manual", "scripted")


class RetryAuditTests(unittest.TestCase):
    def test_stable_key_retry_after_timeout_is_justified(self):
        row = run_actions(case_for("lost_ack"), ["k", "k"])
        self.assertEqual(retry_audit(row), {"uncertain_writes": 1, "retry_decisions": 1,
                                            "justified_retries": 1, "unjustified_retries": 0})
        self.assertTrue(row["success"])

    def test_retry_with_new_or_missing_key_is_unjustified_even_when_it_succeeds(self):
        for keys in (["k1", "k2"], [None, None], [None, "k"]):
            with self.subTest(keys=keys):
                row = run_actions(case_for("pre_commit_timeout"), keys)
                audit = retry_audit(row)
                self.assertEqual((audit["retry_decisions"], audit["unjustified_retries"]), (1, 1))
                self.assertTrue(row["success"])  # pre-commit: the gamble happened to be safe

    def test_key_reuse_is_not_a_guarantee_without_advertised_idempotency(self):
        row = run_actions(case_for("pre_commit_timeout"), ["k", "k"], idempotency=False)
        self.assertEqual(retry_audit(row)["unjustified_retries"], 1)

    def test_retry_after_an_error_that_proves_non_execution_is_not_a_decision(self):
        row = run_actions(case_for("transient_rate_limit"), [None, None])
        self.assertEqual(retry_audit(row)["retry_decisions"], 0)

    def test_rows_without_a_public_trace_are_skipped(self):
        self.assertIsNone(retry_audit({"success": True}))

    def test_policy_summary_separates_safe_from_lucky_success(self):
        cases = build_matrix(workflows="ticket_dispatch", variants=1, seeds=2, faults="pre_commit_timeout")
        rows = [run_scripted(c, p) for c in cases for p in ("idempotent_retry", "bounded_retry")]
        by_agent = {r["agent_id"]: r for r in retry_permission(rows) if r["fault"] == "all"}
        safe, lucky = by_agent["idempotent_retry"], by_agent["bounded_retry"]
        self.assertEqual(safe["justified_retry_share"], 1.0)
        self.assertEqual(safe["success_without_unjustified_retry_rate"], 1.0)
        self.assertEqual(lucky["successes"], 2)
        self.assertEqual(lucky["successes_with_unjustified_retry"], 2)
        self.assertEqual(lucky["success_without_unjustified_retry_rate"], 0.0)
        self.assertIsNone(next(r for r in retry_permission(
            [run_scripted(cases[0], "no_retry")]) if r["fault"] == "all")["justified_retry_share"])


class ConditionalRecoveryTests(unittest.TestCase):
    def test_only_cases_with_a_successful_clean_pair_are_eligible(self):
        cases = build_matrix(workflows="ticket_dispatch", variants=1, seeds=2,
                             faults=["clean", "lost_ack", "permanent_failure"])
        rows = [run_scripted(c, "idempotent_retry") for c in cases]
        clean_seed1 = next(r for r in rows if r["fault"] == "clean" and r["seed"] == 1)
        clean_seed1["success"] = False  # seed 1 is no longer eligible
        result = {c["fault"]: c for c in conditional_recovery(rows)}
        self.assertEqual(result["lost_ack"]["faulted_episodes"], 2)
        self.assertEqual(result["lost_ack"]["eligible_episodes"], 1)
        self.assertEqual(result["lost_ack"]["conditional_recovery_success_rate"], 1.0)
        self.assertEqual(result["permanent_failure"]["conditional_recovery_success_rate"], 0.0)
        # permanent failure is unreachable by design and stays out of the aggregate
        self.assertEqual(result["recoverable_faults"]["eligible_episodes"], 1)
        self.assertEqual(result["recoverable_faults"]["conditional_recovery_success_rate"], 1.0)

    def test_no_clean_success_gives_no_rate(self):
        rows = [run_scripted(case_for("lost_ack"), "no_retry")]
        result = conditional_recovery(rows)[0]
        self.assertEqual(result["eligible_episodes"], 0)
        self.assertIsNone(result["conditional_recovery_success_rate"])

    def test_summary_and_report_include_both_metrics(self):
        cases = build_matrix(workflows="ticket_dispatch", variants=1, seeds=1, faults=["clean", "lost_ack"])
        rows = [run_scripted(c, p) for c in cases for p in ("idempotent_retry", "bounded_retry")]
        summary = summarize(rows)
        self.assertIn("conditional_recovery", summary)
        self.assertIn("retry_permission", summary)
        with tempfile.TemporaryDirectory() as directory:
            finalize_run(rows, Path(directory) / "run")
            page = (Path(directory) / "run" / "report.html").read_text()
        self.assertIn("条件恢复与重试依据", page)
        self.assertIn("100.0% (1/1)", page)


if __name__ == "__main__":
    unittest.main()
