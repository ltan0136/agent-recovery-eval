"""Behavioral checks for recovery scenarios and the evaluator boundary."""

from copy import deepcopy
import unittest

from recovery_eval.core import (
    FAULTS,
    POLICIES,
    RecoveryEnvironment,
    build_cases,
    run_case,
)


def make_case(fault="clean", target_op="create_record", delay=6):
    case = build_cases(task_count=1, seeds=1)[0]
    case.update(
        case_id=f"test-{fault}-{target_op}",
        fault=fault,
        target_op=target_op,
        late_commit_delay=delay,
    )
    return case


def record_args(task):
    return {key: task[key] for key in ("task_id", "title", "body")}


def notification_args(task, record_id):
    return {
        "task_id": task["task_id"],
        "record_id": record_id,
        "recipient": task["recipient"],
        "message": task["message"],
    }


class ScriptedPolicyTests(unittest.TestCase):
    def test_clean_tasks_succeed_under_every_policy(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = run_case(make_case(), policy)
                self.assertTrue(result["success"])
                self.assertEqual(result["duplicates"], 0)
                self.assertEqual(result["missing"], 0)
                self.assertFalse(result["false_success"])
                self.assertEqual(len(result["ledger"]), 2)

    def test_retry_after_lost_ack_is_detected_as_duplicate(self):
        for target in ("create_record", "send_notification"):
            with self.subTest(target=target):
                result = run_case(make_case("ack_lost", target), "bounded_retry")
                self.assertTrue(result["claimed_success"])
                self.assertFalse(result["success"])
                self.assertTrue(result["false_success"])
                self.assertEqual(result["duplicates"], 1)
                self.assertEqual(result["missing"], 0)

    def test_settling_late_writes_exposes_duplicates(self):
        for target in ("create_record", "send_notification"):
            with self.subTest(target=target):
                env = RecoveryEnvironment(make_case("late_commit", target, delay=6))
                task = env.task
                if target == "create_record":
                    first = env.invoke("create_record", record_args(task))
                    self.assertEqual(first["error"]["type"], "timeout")
                    second = env.invoke("create_record", record_args(task))
                    record_id = second["result"]["record_id"]
                    env.invoke("send_notification", notification_args(task, record_id))
                else:
                    created = env.invoke("create_record", record_args(task))
                    args = notification_args(task, created["result"]["record_id"])
                    first = env.invoke("send_notification", args)
                    self.assertEqual(first["error"]["type"], "timeout")
                    env.invoke("send_notification", args)
                self.assertTrue(env.grade()["success"])
                result = env.finish(claimed_success=True)
                self.assertFalse(result["success"])
                self.assertEqual(result["duplicates"], 1)
                self.assertTrue(result["false_success"])
                self.assertTrue(any(row["source"] == "late_delivery" for row in result["ledger"]))

    def test_single_status_check_is_insufficient_with_stale_reads(self):
        for target in ("create_record", "send_notification"):
            with self.subTest(target=target):
                fresh = run_case(make_case("ack_lost", target), "verify_then_retry")
                stale = run_case(make_case("stale_read", target), "verify_then_retry")
                self.assertTrue(fresh["success"])
                self.assertFalse(stale["success"])
                self.assertEqual(stale["duplicates"], 1)
                self.assertTrue(any(event["event"] == "stale_snapshot_served" for event in stale["trace"]))

    def test_idempotency_handles_each_supported_fault_and_both_operations(self):
        for fault in FAULTS:
            for target in ("create_record", "send_notification"):
                for delay in (4, 6):
                    with self.subTest(fault=fault, target=target, delay=delay):
                        result = run_case(make_case(fault, target, delay), "idempotent_retry")
                        self.assertTrue(result["success"])
                        self.assertEqual(len(result["ledger"]), 2)
                        record = result["final_state"]["records"][0]
                        notification = result["final_state"]["notifications"][0]
                        self.assertEqual(notification["record_id"], record["record_id"])
                        self.assertEqual(record["title"], make_case()["payload"]["title"])


class EvaluatorInvariantTests(unittest.TestCase):
    def test_empty_episode_is_two_missing_effects(self):
        env = RecoveryEnvironment(make_case())
        result = env.finish(claimed_success=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["missing"], 2)
        self.assertEqual(result["duplicates"], 0)
        self.assertTrue(result["false_success"])

    def test_wrong_record_content_is_not_success(self):
        env = RecoveryEnvironment(make_case())
        task = env.task
        args = record_args(task)
        args["body"] = "An unrelated request"
        created = env.invoke("create_record", args)
        env.invoke("send_notification", notification_args(task, created["result"]["record_id"]))
        score = env.grade()
        self.assertFalse(score["success"])
        self.assertGreater(score["wrong_payload"], 0)

    def test_notification_linking_to_nonexistent_record_is_not_success(self):
        env = RecoveryEnvironment(make_case())
        task = env.task
        env.invoke("create_record", record_args(task))
        env.invoke("send_notification", notification_args(task, "record-does-not-exist"))
        score = env.grade()
        self.assertFalse(score["success"])
        self.assertEqual(score["wrong_payload"], 1)
        self.assertEqual(score["missing"], 0)

    def test_grade_and_snapshot_do_not_advance_or_mutate_episode(self):
        inspected = RecoveryEnvironment(make_case("late_commit", delay=6))
        control = RecoveryEnvironment(make_case("late_commit", delay=6))
        for env in (inspected, control):
            response = env.invoke("create_record", record_args(env.task))
            self.assertEqual(response["error"]["type"], "timeout")
        snapshot_before = inspected.snapshot()
        for _ in range(5):
            score = inspected.grade()
            self.assertEqual(inspected.snapshot(), snapshot_before)
            self.assertEqual(score["missing"], 2)
        # Identical final traces also catch hidden tick advances or ledger writes.
        self.assertEqual(inspected.finish(False), control.finish(False))

    def test_public_task_and_returned_snapshots_are_detached_copies(self):
        env = RecoveryEnvironment(make_case())
        task = env.task
        original_title = task["title"]
        task["title"] = "Changed outside the environment"
        self.assertEqual(env.task["title"], original_title)
        env.invoke("create_record", record_args(env.task))
        snapshot = env.snapshot()
        snapshot["records"][0]["title"] = "Changed snapshot"
        self.assertEqual(env.snapshot()["records"][0]["title"], original_title)

    def test_cases_have_isolated_state_and_fault_schedule(self):
        case = make_case("before_commit_once")
        first = RecoveryEnvironment(case)
        second = RecoveryEnvironment(case)
        args = record_args(first.task)
        self.assertEqual(first.invoke("create_record", args)["error"]["type"], "timeout")
        self.assertTrue(first.invoke("create_record", args)["ok"])
        self.assertEqual(second.snapshot(), {"records": [], "notifications": []})
        self.assertEqual(second.invoke("create_record", args)["error"]["type"], "timeout")
        self.assertEqual(second.snapshot(), {"records": [], "notifications": []})

    def test_same_inputs_produce_identical_results_without_mutating_case(self):
        case = make_case("late_commit", "send_notification")
        untouched = deepcopy(case)
        first = run_case(case, "verify_then_retry")
        second = run_case(case, "verify_then_retry")
        self.assertEqual(first, second)
        self.assertEqual(case, untouched)

    def test_idempotency_key_cannot_be_reused_with_different_payload(self):
        for fault in ("clean", "late_commit"):
            with self.subTest(fault=fault):
                env = RecoveryEnvironment(make_case(fault))
                args = {**record_args(env.task), "operation_id": "same-request"}
                env.invoke("create_record", args)
                changed = {**args, "title": "Conflicting title"}
                response = env.invoke("create_record", changed)
                self.assertFalse(response["ok"])
                self.assertEqual(response["error"]["type"], "tool_error")
                result = env.finish(False)
                self.assertEqual(len(result["ledger"]), 1)
                self.assertEqual(result["ledger"][0]["payload"]["title"], args["title"])

    def test_public_observations_do_not_reveal_fault_or_committed_state(self):
        before = RecoveryEnvironment(make_case("before_commit_once"))
        after = RecoveryEnvironment(make_case("ack_lost"))
        self.assertEqual(before.task, after.task)
        self.assertEqual(set(before.task), {"task_id", "title", "body", "recipient", "message"})
        response_before = before.invoke("create_record", record_args(before.task))
        response_after = after.invoke("create_record", record_args(after.task))
        self.assertEqual(response_before, response_after)
        self.assertEqual(response_before, {
            "ok": False,
            "error": {"type": "timeout", "message": "request timed out; outcome unknown"},
        })
        self.assertNotEqual(before.snapshot(), after.snapshot())

    def test_finishing_closes_tools_and_prevents_repeated_finish(self):
        env = RecoveryEnvironment(make_case())
        env.finish(False)
        response = env.invoke("create_record", record_args(env.task))
        self.assertEqual(response["error"]["type"], "closed")
        self.assertEqual(env.snapshot(), {"records": [], "notifications": []})
        with self.assertRaises(RuntimeError):
            env.finish(False)

    def test_seeded_schedules_are_reproducible_and_actually_vary(self):
        cases = build_cases(task_count=2, seeds=20)
        self.assertEqual(cases, build_cases(task_count=2, seeds=20))
        self.assertEqual(len({case["case_id"] for case in cases}), len(cases))
        one_fault = [case for case in cases if case["fault"] == "late_commit"]
        self.assertEqual({case["target_op"] for case in one_fault}, {"create_record", "send_notification"})
        self.assertGreater(len({case["late_commit_delay"] for case in one_fault}), 1)
        signatures = set()
        for case in one_fault:
            result = run_case(case, "bounded_retry")
            scheduled = next(event for event in result["trace"] if event["event"] == "write_scheduled")
            signatures.add((scheduled["operation"], scheduled["due_tick"] - scheduled["tick"]))
        self.assertGreater(len(signatures), 2)


if __name__ == "__main__":
    unittest.main()
