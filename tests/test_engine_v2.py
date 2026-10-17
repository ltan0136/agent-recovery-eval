"""Independent business invariants and public-contract tests for the v2 engine."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import unittest

from recovery_eval.engine import Environment, run_scripted
from recovery_eval.workflows import FAULTS, POLICIES, build_matrix, config_hash


def case_for(workflow="ticket_dispatch", fault="clean", target=None, seed=0):
    cases = build_matrix([workflow], variants=1, seeds=seed + 1, faults=[fault])
    case = next(case for case in cases if case["seed"] == seed)
    if target is not None:
        case["target_step"] = target
        case["config_hash"] = config_hash(case)
    return case


def execute_plan(environment, order=None, stable_keys=False):
    plan = environment.public_task["steps"]
    by_name = {step["name"]: step for step in plan}
    responses = []
    for name in order or [step["name"] for step in plan]:
        step = by_name[name]
        arguments = deepcopy(step["arguments"])
        if stable_keys:
            arguments["operation_id"] = f"test:{name}"
        responses.append(environment.invoke(step["tool"], arguments))
    return responses


class MatrixContractTests(unittest.TestCase):
    def test_default_matrix_has_unique_cases_and_only_sixteen_business_variants(self):
        cases = build_matrix()
        self.assertEqual(len(cases), 384)
        self.assertEqual(len({case["case_id"] for case in cases}), 384)
        self.assertEqual({case["fault"] for case in cases}, set(FAULTS))
        self.assertEqual(len({(case["workflow"], case["variant_id"]) for case in cases}), 16)
        for case in cases:
            self.assertEqual(case["config_hash"], config_hash(case))

    def test_seed_changes_a_real_schedule_and_target_without_changing_public_goal(self):
        first = case_for(fault="late_commit", seed=0)
        second = case_for(fault="late_commit", seed=1)
        self.assertNotEqual(first["target_step"], second["target_step"])
        self.assertNotEqual(first["late_commit_delay"], second["late_commit_delay"])
        self.assertEqual(first["variant_id"], second["variant_id"])
        self.assertEqual(Environment(first).public_task, Environment(second).public_task)

    def test_matrix_rejects_invalid_counts_unknown_conditions_and_duplicate_selection(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_matrix(variants=value)
        for arguments in ({"faults": ["unknown"]}, {"workflows": ["unknown"]},
                          {"faults": ["clean", "clean"]}, {"workflows": []}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                build_matrix(**arguments)

    def test_public_task_and_schemas_contain_no_hidden_condition_fields(self):
        public = Environment(case_for(fault="stale_read", seed=2)).public_task
        hidden = {"fault", "target_step", "seed", "config_hash", "condition_hash", "case_config", "final_state"}

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(hidden & set(value))
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(public)
        check(Environment(case_for()).tool_schemas)
        self.assertTrue(public["plan_provided"])
        self.assertEqual(public["workflow_id"], "ticket_dispatch")
        self.assertEqual(public["steps"][0]["arguments"]["ticket_ref"], "T-001")


class DomainInvariantTests(unittest.TestCase):
    def test_all_domains_complete_with_their_distinct_business_state(self):
        results = {}
        for workflow in ("ticket_dispatch", "inventory_order", "meeting_booking", "document_publication"):
            result = run_scripted(case_for(workflow), "no_retry")
            self.assertTrue(result["success"], workflow)
            self.assertEqual(result["duplicates"], 0)
            self.assertEqual(result["missing"], 0)
            results[workflow] = result["final_state"]
        ticket = results["ticket_dispatch"]["tickets"][0]
        self.assertEqual(ticket["status"], "resolved")
        self.assertEqual(ticket["owner"], "engineer-1")
        self.assertEqual(results["ticket_dispatch"]["notifications"][0]["recipient"], "requester-1@example.test")
        inventory = results["inventory_order"]
        self.assertEqual(inventory["initial_stock"] - inventory["available_stock"], 1)
        self.assertEqual(sum(charge["amount_cents"] for charge in inventory["charges"]), 1075)
        self.assertEqual(inventory["orders"][0]["status"], "shipped")
        meeting = results["meeting_booking"]
        self.assertEqual(len(meeting["holds"]), 1)
        self.assertEqual(len(meeting["invitations"][0]["attendees"]), 2)
        self.assertEqual(meeting["confirmations"][0]["hold_id"], meeting["holds"][0]["id"])
        document = results["document_publication"]
        self.assertTrue(document["checks"][0]["passed"])
        self.assertEqual(document["publications"][0]["approval_id"], document["approvals"][0]["id"])
        self.assertEqual(document["documents"][0]["revision"], document["publications"][0]["revision"])

    def test_inventory_allows_payment_before_reservation_but_not_shipment(self):
        environment = Environment(case_for("inventory_order"))
        plan = {step["name"]: step for step in environment.public_task["steps"]}
        for name in ("order", "charge"):
            self.assertTrue(environment.invoke(plan[name]["tool"], plan[name]["arguments"])["ok"])
        rejected = environment.invoke(plan["ship"]["tool"], plan["ship"]["arguments"])
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["error"]["type"], "prerequisite_not_committed")
        self.assertTrue(all(response["ok"] for response in execute_plan(environment, ["reserve", "ship", "receipt"])))
        result = environment.finish(True)
        self.assertTrue(result["success"])
        self.assertEqual([item["step"] for item in result["ledger"]], ["order", "charge", "reserve", "ship", "receipt"])

    def test_meeting_can_invite_before_room_hold_but_cannot_confirm(self):
        environment = Environment(case_for("meeting_booking"))
        self.assertTrue(all(response["ok"] for response in execute_plan(environment, ["meeting", "invite"])))
        plan = {step["name"]: step for step in environment.public_task["steps"]}
        self.assertFalse(environment.invoke(plan["confirm"]["tool"], plan["confirm"]["arguments"])["ok"])
        self.assertTrue(all(response["ok"] for response in execute_plan(environment, ["hold", "confirm"])))
        self.assertTrue(environment.finish(True)["success"])

    def test_server_blocks_publication_before_matching_approval(self):
        environment = Environment(case_for("document_publication"))
        self.assertTrue(all(response["ok"] for response in execute_plan(environment, ["submit", "check"])))
        plan = {step["name"]: step for step in environment.public_task["steps"]}
        self.assertFalse(environment.invoke(plan["publish"]["tool"], plan["publish"]["arguments"])["ok"])
        wrong_revision = {**plan["approve"]["arguments"], "revision": 999}
        rejected = environment.invoke(plan["approve"]["tool"], wrong_revision)
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["error"]["type"], "business_constraint")
        self.assertTrue(all(response["ok"] for response in execute_plan(environment, ["approve", "publish", "notify"])))
        self.assertTrue(environment.finish(True)["success"])

    def test_syntactically_valid_wrong_recipient_is_rejected_by_independent_oracle(self):
        environment = Environment(case_for())
        execute_plan(environment, ["create", "assign", "resolve"])
        notification = environment.public_task["steps"][-1]
        arguments = {**notification["arguments"], "recipient": "wrong@example.test"}
        self.assertTrue(environment.invoke(notification["tool"], arguments)["ok"])
        result = environment.finish(True)
        self.assertFalse(result["success"])
        self.assertTrue(result["task_complete"])
        self.assertEqual(result["wrong_payload"], 1)
        self.assertTrue(result["false_success"])

    def test_wrong_resource_reference_cannot_silently_associate_another_order(self):
        environment = Environment(case_for("inventory_order"))
        execute_plan(environment, ["reserve", "order", "charge"])
        shipment = next(step for step in environment.public_task["steps"] if step["name"] == "ship")
        response = environment.invoke(shipment["tool"], {**shipment["arguments"], "order_ref": "O-NOT-EXISTING"})
        self.assertFalse(response["ok"])
        result = environment.finish(True)
        self.assertFalse(result["success"])
        self.assertEqual(result["final_state"]["shipments"], [])
        self.assertEqual(result["missing_steps"], ["ship", "receipt"])

    def test_room_capacity_prevents_duplicate_hold_effect_without_assuming_episode_failure(self):
        result = run_scripted(case_for("meeting_booking", "duplicate_delivery", "hold"), "bounded_retry")
        self.assertEqual(len(result["final_state"]["holds"]), 1)
        self.assertEqual(result["resource_violations"], 0)
        self.assertEqual(result["duplicates"], 0)
        self.assertTrue(result["success"])


class RecoveryBehaviorTests(unittest.TestCase):
    def test_permanent_failure_measures_missing_work_and_false_claims(self):
        case = case_for(fault="permanent_failure", target="assign")
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = run_scripted(case, policy)
                self.assertFalse(result["success"])
                self.assertFalse(result["claimed_success"])
                self.assertEqual(result["missing_steps"], ["assign", "resolve", "notify"])
        environment = Environment(case)
        execute_plan(environment)
        self.assertTrue(environment.finish(True)["false_success"])

    def test_timeout_outcomes_are_indistinguishable_to_caller(self):
        responses = []
        for fault in ("pre_commit_timeout", "lost_ack", "late_commit"):
            environment = Environment(case_for(fault=fault, target="create"))
            step = environment.public_task["steps"][0]
            responses.append(environment.invoke(step["tool"], step["arguments"]))
        self.assertEqual(responses[0], responses[1])
        self.assertEqual(responses[1], responses[2])

    def test_rate_limit_and_permanent_failure_have_distinct_retry_contracts(self):
        responses = {}
        for fault in ("transient_rate_limit", "permanent_failure"):
            environment = Environment(case_for(fault=fault, target="create"))
            step = environment.public_task["steps"][0]
            responses[fault] = environment.invoke(step["tool"], step["arguments"])
        self.assertEqual(responses["transient_rate_limit"]["error"]["type"], "rate_limited")
        self.assertTrue(responses["transient_rate_limit"]["error"]["retryable"])
        self.assertFalse(responses["permanent_failure"]["error"]["retryable"])

    def test_lost_ack_duplicate_retry_is_real_and_stable_key_prevents_it(self):
        case = case_for(fault="lost_ack", target="create")
        bounded = run_scripted(case, "bounded_retry")
        stable = run_scripted(case, "idempotent_retry")
        self.assertFalse(bounded["success"])
        self.assertEqual(len(bounded["final_state"]["tickets"]), 2)
        self.assertEqual(bounded["duplicates"], 1)
        self.assertTrue(stable["success"])
        self.assertEqual(len(stable["final_state"]["tickets"]), 1)

    def test_stale_read_does_not_make_verification_a_safe_retry_protocol(self):
        case = case_for(fault="stale_read", target="create")
        result = run_scripted(case, "verify_then_retry")
        self.assertFalse(result["success"])
        self.assertGreater(result["duplicates"], 0)
        self.assertTrue(any(item["tool"] == "inspect_step" for item in result["trace"] if item["event"] == "tool_call"))

    def test_settle_includes_a_final_late_write_before_readonly_grading(self):
        case = case_for(fault="late_commit", target="notify")
        environment = Environment(case)
        responses = execute_plan(environment)
        self.assertFalse(responses[-1]["ok"])
        self.assertEqual(len(environment.ledger), 3)
        result = environment.finish(False)
        self.assertTrue(result["success"])
        self.assertTrue(result["false_failure"])
        self.assertFalse(result["false_success"])
        self.assertEqual(len(result["ledger"]), 4)
        self.assertEqual(len(result["final_state"]["notifications"]), 1)
        self.assertTrue(result["oracle_read_only"])
        self.assertTrue(result["settled_before_grade"])
        self.assertEqual(environment.finish(True), result)
        self.assertEqual(environment.ledger, result["ledger"])
        self.assertEqual(environment.trace, result["trace"])

    def test_budget_stops_further_effects_and_records_missing_work(self):
        environment = Environment(case_for(), max_calls=1)
        plan = environment.public_task["steps"]
        self.assertTrue(environment.invoke(plan[0]["tool"], plan[0]["arguments"])["ok"])
        self.assertEqual(environment.invoke(plan[1]["tool"], plan[1]["arguments"])["error"]["type"], "budget_exhausted")
        result = environment.finish(True)
        self.assertEqual(result["tool_calls"], 1)
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(len(result["ledger"]), 1)
        self.assertEqual(result["missing"], 3)
        self.assertTrue(result["false_success"])

    def test_unsupported_idempotency_can_degrade_even_if_policy_supplies_keys(self):
        case = case_for(fault="lost_ack", target="create")
        supported = run_scripted(case, "idempotent_retry", idempotency=True)
        unsupported = run_scripted(case, "idempotent_retry", idempotency=False)
        self.assertTrue(supported["success"])
        self.assertFalse(unsupported["success"])
        self.assertEqual(unsupported["duplicates"], 1)
        self.assertFalse(unsupported["public_task"]["service_contract"]["idempotency_supported"])

    def test_key_cannot_be_reused_for_a_different_payload(self):
        environment = Environment(case_for())
        step = environment.public_task["steps"][0]
        arguments = {**step["arguments"], "operation_id": "stable-create"}
        self.assertTrue(environment.invoke(step["tool"], arguments)["ok"])
        response = environment.invoke(step["tool"], {**arguments, "title": "Changed title"})
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["type"], "idempotency_conflict")
        self.assertEqual(len(environment.ledger), 1)


class ReproducibilityAndOracleTests(unittest.TestCase):
    def test_malformed_inspection_steps_and_tool_names_return_errors_without_crashing(self):
        environment = Environment(case_for())
        for value in ([], {}, None, 1, True):
            with self.subTest(step=value):
                response = environment.invoke("inspect_step", {"step": value})
                self.assertFalse(response["ok"])
                self.assertEqual(response["error"]["type"], "invalid_arguments")
        for value in ([], {}, None, 1):
            with self.subTest(tool=value):
                response = environment.invoke(value, {})
                self.assertFalse(response["ok"])
                self.assertEqual(response["error"]["type"], "invalid_tool_name")
        self.assertEqual(environment.ledger, [])
        self.assertEqual(environment.finish(False)["tool_calls"], 9)

    def test_nested_or_nonstring_array_items_cannot_crash_business_constraints(self):
        for workflow, preparatory_steps, tool, field in (
            ("meeting_booking", ["meeting"], "invite_attendees", "attendees"),
            ("document_publication", ["submit"], "check_document", "checklist"),
        ):
            environment = Environment(case_for(workflow))
            execute_plan(environment, preparatory_steps)
            step = next(step for step in environment.public_task["steps"] if step["tool"] == tool)
            before = environment.ledger
            for value in ([[]], [{}], [None], [1], [True], ["valid", []]):
                with self.subTest(workflow=workflow, value=value):
                    response = environment.invoke(tool, {**step["arguments"], field: value})
                    self.assertFalse(response["ok"])
                    self.assertEqual(response["error"]["type"], "invalid_arguments")
                    self.assertEqual(environment.ledger, before)

    def test_condition_hash_covers_entire_execution_configuration(self):
        case = case_for(fault="late_commit")
        baseline = Environment(case).finish(False)
        expected = hashlib.sha256(json.dumps(baseline["execution_config"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        self.assertEqual(baseline["condition_hash"], expected)
        self.assertEqual(baseline["config_hash"], expected)
        modified = deepcopy(case)
        modified["late_commit_delay"] += 1
        variants = [Environment(modified).finish(False), Environment(case, max_calls=23).finish(False),
                    Environment(case, idempotency=False).finish(False)]
        for result in variants:
            self.assertNotEqual(result["condition_hash"], baseline["condition_hash"])
        self.assertIn("workflow_plan", baseline["execution_config"])

    def test_episodes_are_isolated_and_public_snapshots_cannot_mutate_environment(self):
        case = case_for()
        first = Environment(case)
        second = Environment(case)
        step = first.public_task["steps"][0]
        first.invoke(step["tool"], step["arguments"])
        observed = second.invoke("inspect_step", {"step": "create"})
        self.assertFalse(observed["result"]["completed"])
        returned_ledger = first.ledger
        returned_ledger[0]["arguments"]["title"] = "Tampered"
        public = first.public_task
        public["params"]["title"] = "Tampered"
        self.assertEqual(first.ledger[0]["arguments"]["title"], "Restore service 1")
        self.assertEqual(first.public_task["params"]["title"], "Restore service 1")
        self.assertEqual(second.finish(False)["ledger"], [])

    def test_result_snapshots_and_repeated_finish_do_not_mutate_truth(self):
        environment = Environment(case_for())
        execute_plan(environment)
        result = environment.finish(True, "test", "scripted")
        result["final_state"]["tickets"].clear()
        result["ledger"].clear()
        self.assertEqual(len(environment.finish(False)["final_state"]["tickets"]), 1)
        self.assertEqual(len(environment.finish(False)["ledger"]), 4)
        self.assertEqual(environment.invoke("inspect_step", {"step": "create"})["error"]["type"], "finished")

    def test_finish_metadata_cannot_override_the_independent_score(self):
        for metadata in ({"success": True}, {"duplicates": 0}, {"case_config": {}}, {"condition_hash": "forged"}):
            environment = Environment(case_for())
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                environment.finish(True, **metadata)
        result = Environment(case_for()).finish(False, experiment_label="unit-test")
        self.assertEqual(result["experiment_label"], "unit-test")
        self.assertFalse(result["success"])

    def test_deterministic_result_and_complete_json_serialization(self):
        case = case_for("document_publication", "late_commit", "approve", seed=2)
        first = run_scripted(case, "idempotent_retry")
        second = run_scripted(deepcopy(case), "idempotent_retry")
        self.assertEqual(first, second)
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)


if __name__ == "__main__":
    unittest.main()
