"""Fault injection into the v2 judge: every business check must be able to fail.

The simulated service rejects most invalid writes before they reach the
ledger, so tests that only drive the public tools never show the judge most of
the states its checks exist for. Two kinds of case live here:

* Reachable: a retry after a lost acknowledgement really produces the
  violation through the public tools.
* Injected: a correct episode is settled through the public tools, then its
  private state or ledger is corrupted in one way, as if the service had a bug,
  and the judge must name that exact problem.

Expected failure reasons are written by hand from the business rule; none is
obtained by calling the judge on another input. scripts/oracle_mutation_check.py
disables each judge check in turn and confirms that a test here fails.
"""

from __future__ import annotations

from copy import deepcopy
import unittest

from recovery_eval.engine import Environment, run_scripted
from recovery_eval.workflows import build_matrix, config_hash, workflow_specs


def case_for(workflow: str, fault: str = "clean", target: str | None = None) -> dict:
    case = build_matrix([workflow], variants=1, seeds=1, faults=[fault])[0]
    if target is not None:
        case["target_step"] = target
        case["config_hash"] = config_hash(case)
    return case


class JudgeAssertions(unittest.TestCase):
    def settled(self, workflow: str) -> Environment:
        """Execute the public plan once on a clean case; every step must commit."""
        environment = Environment(case_for(workflow))
        for step in environment.public_task["steps"]:
            response = environment.invoke(step["tool"], step["arguments"])
            self.assertTrue(response["ok"], response)
        return environment

    def assertJudged(self, result: dict, reasons: list[str]) -> None:
        """The episode fails for exactly these reasons, whatever the agent claims."""
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reasons"], reasons)
        if result["claimed_success"]:
            self.assertTrue(result["false_success"])


class SettledControlTests(JudgeAssertions):
    def test_uncorrupted_settled_episodes_pass_every_check(self):
        # Control for the injected cases below: the corruption, not the
        # settled episode itself, is what makes the judge fail them.
        for workflow in workflow_specs():
            with self.subTest(workflow=workflow):
                result = self.settled(workflow).finish(True)
                self.assertTrue(result["success"])
                self.assertEqual(result["failure_reasons"], [])
                self.assertEqual(result["resource_violations"], 0)
                self.assertEqual(result["order_violations"], 0)


class ReachableViolationTests(JudgeAssertions):
    """Violations a blind retry after a lost acknowledgement really produces."""

    def test_retried_reservation_after_lost_ack_reserves_stock_twice(self):
        case = case_for("inventory_order", "lost_ack", "reserve")
        result = run_scripted(case, "bounded_retry")
        reservations = result["final_state"]["reservations"]
        self.assertEqual(len(reservations), 2)
        self.assertEqual(sum(item["quantity"] for item in reservations), 2 * case["params"]["quantity"])
        self.assertEqual(result["resource_violations"], 1)
        self.assertJudged(result, ["duplicate_effects", "resource_violations"])

    def test_retried_charge_after_lost_ack_bills_the_customer_twice(self):
        case = case_for("inventory_order", "lost_ack", "charge")
        result = run_scripted(case, "bounded_retry")
        charges = result["final_state"]["charges"]
        self.assertEqual(len(charges), 2)
        self.assertEqual(sum(item["amount_cents"] for item in charges), 2 * case["params"]["amount_cents"])
        self.assertEqual(result["resource_violations"], 1)
        self.assertJudged(result, ["duplicate_effects", "resource_violations"])

    def test_retried_invitation_after_lost_ack_leaves_two_invitation_batches(self):
        case = case_for("meeting_booking", "lost_ack", "invite")
        result = run_scripted(case, "bounded_retry")
        self.assertEqual(len(result["final_state"]["invitations"]), 2)
        self.assertEqual(len(result["final_state"]["confirmations"]), 1)
        self.assertJudged(result, ["duplicate_effects", "confirmation_requires_one_invitation_batch"])

    def test_wrong_recipient_fails_both_the_payload_and_the_state_content_check(self):
        environment = Environment(case_for("ticket_dispatch"))
        steps = environment.public_task["steps"]
        for step in steps[:-1]:
            self.assertTrue(environment.invoke(step["tool"], step["arguments"])["ok"])
        notify = steps[-1]
        self.assertTrue(environment.invoke(notify["tool"], {**notify["arguments"],
                                                            "recipient": "wrong@example.test"})["ok"])
        self.assertJudged(environment.finish(True), ["wrong_payload", "notifications_content_mismatch"])


class InjectedStateTests(JudgeAssertions):
    """Settled states the service would never produce, fed straight to the judge.

    The private _state and _ledger attributes are edited on purpose: these
    states cannot be reached through the public tools, which is why the
    judge's own checks need a test that does not go through them.
    """

    def test_ledger_out_of_dependency_order_is_flagged(self):
        environment = self.settled("ticket_dispatch")
        ledger = environment._ledger
        # The assignment now appears to commit before its ticket existed.
        ledger[0], ledger[1] = ledger[1], ledger[0]
        result = environment.finish(True)
        self.assertEqual(result["order_violations"], 1)
        self.assertJudged(result, ["order_violations"])

    def test_side_effect_missing_from_the_ledger_is_flagged(self):
        environment = self.settled("ticket_dispatch")
        notifications = environment._state["notifications"]
        # A second notification reached the requester but was never recorded.
        notifications.append({**deepcopy(notifications[0]), "id": "notifications-002"})
        result = environment.finish(True)
        # The duplicate counter reads the ledger, so only the state/ledger
        # cross-check can see this extra side effect.
        self.assertEqual(result["duplicates"], 0)
        self.assertJudged(result, ["notifications_state_ledger_mismatch"])

    def test_ticket_not_left_resolved_is_flagged(self):
        environment = self.settled("ticket_dispatch")
        environment._state["tickets"][0]["status"] = "assigned"
        self.assertJudged(environment.finish(True), ["ticket_final_state_mismatch"])

    def test_notification_for_an_unknown_ticket_is_flagged(self):
        environment = self.settled("ticket_dispatch")
        environment._state["notifications"][0]["ticket_id"] = "tickets-999"
        self.assertJudged(environment.finish(True), ["broken_ticket_reference"])

    def test_reservation_without_a_stock_decrement_is_flagged(self):
        environment = self.settled("inventory_order")
        state = environment._state
        # Stock was reserved but the available count was never reduced.
        state["available_stock"] = state["initial_stock"]
        self.assertJudged(environment.finish(True), ["inventory_conservation_failure"])

    def test_shipped_order_left_in_paid_status_is_flagged(self):
        environment = self.settled("inventory_order")
        environment._state["orders"][0]["status"] = "paid"
        self.assertJudged(environment.finish(True), ["order_final_state_mismatch"])

    def test_shipment_citing_an_unknown_reservation_is_flagged(self):
        environment = self.settled("inventory_order")
        environment._state["shipments"][0]["reservation_id"] = "reservations-999"
        self.assertJudged(environment.finish(True), ["shipment_resource_reference_mismatch"])

    def test_double_booked_room_slot_is_flagged(self):
        environment = self.settled("meeting_booking")
        state, ledger = environment._state, environment._ledger
        # As if the service's capacity guard had let a second hold through.
        hold_effect = next(effect for effect in ledger if effect["step"] == "hold")
        ledger.append({**deepcopy(hold_effect), "effect_id": "effect-9001"})
        state["holds"].append({**deepcopy(state["holds"][0]), "id": "holds-002"})
        result = environment.finish(True)
        self.assertEqual(result["resource_violations"], 1)
        self.assertJudged(result, ["duplicate_effects", "resource_violations"])

    def test_confirmation_citing_an_unknown_hold_is_flagged(self):
        environment = self.settled("meeting_booking")
        environment._state["confirmations"][0]["hold_id"] = "holds-999"
        self.assertJudged(environment.finish(True), ["confirmation_resource_reference_mismatch"])

    def test_publication_citing_an_unknown_approval_is_flagged(self):
        environment = self.settled("document_publication")
        environment._state["publications"][0]["approval_id"] = "approvals-999"
        self.assertJudged(environment.finish(True), ["publication_without_matching_approval"])

    def test_publication_after_a_failed_document_check_is_flagged(self):
        environment = self.settled("document_publication")
        environment._state["checks"][0]["passed"] = False
        self.assertJudged(environment.finish(True), ["failed_document_checks"])


if __name__ == "__main__":
    unittest.main()
