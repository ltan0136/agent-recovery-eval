"""Exercise live-agent control flow with local model-response mocks only."""

import json
import unittest
from unittest.mock import patch

from recovery_eval.api_agent import run_live_case
from recovery_eval.core import build_cases


def chat_response(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}], "usage": {"total_tokens": 12}}


def tool_call(name, args, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class ApiAgentTests(unittest.TestCase):
    def setUp(self):
        self.case = build_cases(1, 1)[0]
        self.endpoint = "https://unused.example.test/v1"
        self.secret = "private-test-value-do-not-save"

    def run_mocked(self, responses, **kwargs):
        with patch("recovery_eval.api_agent.request_chat", side_effect=responses) as request:
            result = run_live_case(self.case, "mock-model", self.endpoint, self.secret, **kwargs)
        return result, request

    def test_tool_loop_executes_real_environment_and_records_model_claim(self):
        payload = self.case["payload"]
        record = {key: payload[key] for key in ("task_id", "title", "body")}
        notice = {key: payload[key] for key in ("task_id", "recipient", "message")}
        notice["record_id"] = "record-001"
        responses = [
            chat_response(tool_calls=[tool_call("create_record", record)]),
            chat_response(tool_calls=[tool_call("send_notification", notice, "call-2")]),
            chat_response(content='{"success": true}'),
        ]
        result, request = self.run_mocked(responses)
        self.assertEqual(request.call_count, 3)
        self.assertTrue(result["success"])
        self.assertTrue(result["claimed_success"])
        self.assertFalse(result["false_success"])
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(result["evaluation_mode"], "live_model")
        self.assertEqual(len(result["ledger"]), 2)
        self.assertEqual(result["tool_calls"], 2)
        self.assertNotIn(self.secret, json.dumps(result))
        tool_messages = [message for message in result["messages"] if message["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertTrue(all(json.loads(message["content"])["ok"] for message in tool_messages))

    def test_a_success_claim_without_tools_is_graded_as_failure(self):
        result, _ = self.run_mocked([chat_response(content='{"success": true}')])
        self.assertEqual(result["run_status"], "completed")
        self.assertTrue(result["claimed_success"])
        self.assertFalse(result["success"])
        self.assertTrue(result["false_success"])
        self.assertEqual(result["missing"], 2)

    def test_invalid_final_answers_never_become_success_claims(self):
        for content in ("not JSON", '{"success": "true"}', '{"success": true, "extra": 1}', "null", "[]"):
            with self.subTest(content=content):
                result, _ = self.run_mocked([chat_response(content=content)])
                self.assertEqual(result["run_status"], "invalid_final_answer")
                self.assertFalse(result["claimed_success"])
                self.assertFalse(result["success"])

    def test_endpoint_failure_is_marked_separately_from_model_completion(self):
        result, request = self.run_mocked([RuntimeError("network failure")])
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["run_status"], "infrastructure_error")
        self.assertFalse(result["claimed_success"])
        self.assertFalse(result["success"])
        self.assertEqual(result["model_calls"][0]["error_type"], "RuntimeError")
        self.assertNotIn("network failure", json.dumps(result))

    def test_turn_budget_stops_requests_without_fabricating_completion(self):
        responses = [chat_response(tool_calls=[tool_call("wait", {}, f"wait-{i}")]) for i in range(3)]
        result, request = self.run_mocked(responses, max_turns=2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["run_status"], "turn_budget_exhausted")
        self.assertFalse(result["claimed_success"])
        self.assertEqual(result["missing"], 2)
        self.assertEqual(result["logical_ticks"], 2)

    def test_bad_tool_arguments_are_returned_to_model_without_effects(self):
        invalid = tool_call("create_record", {})
        invalid["function"]["arguments"] = "{broken"
        result, _ = self.run_mocked([
            chat_response(tool_calls=[invalid]),
            chat_response(content='{"success": false}'),
        ])
        self.assertEqual(result["run_status"], "completed")
        self.assertFalse(result["claimed_success"])
        self.assertEqual(result["ledger"], [])
        message = next(message for message in result["messages"] if message["role"] == "tool")
        self.assertEqual(json.loads(message["content"])["error"]["type"], "invalid_tool_call")

    def test_malformed_response_envelopes_are_failures_instead_of_crashes(self):
        for response in ({}, {"choices": []}, {"choices": None},
                         {"choices": [{"message": None}]},
                         {"choices": [{"message": "not an object"}]}):
            with self.subTest(response=response):
                result, _ = self.run_mocked([response])
                self.assertNotEqual(result["run_status"], "completed")
                self.assertFalse(result["claimed_success"])
                self.assertEqual(result["ledger"], [])

    def test_missing_tool_call_id_does_not_crash_or_claim_success(self):
        invalid = tool_call("wait", {})
        del invalid["id"]
        result, _ = self.run_mocked([
            chat_response(tool_calls=[invalid]),
            chat_response(content='{"success": false}'),
        ])
        self.assertFalse(result["claimed_success"])
        self.assertFalse(result["success"])
        self.assertEqual(result["ledger"], [])

    def test_oversized_single_tool_batch_is_rejected_before_execution(self):
        response = chat_response(tool_calls=[tool_call("wait", {}, f"wait-{i}") for i in range(25)])
        result, request = self.run_mocked([response])
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["run_status"], "tool_budget_exhausted")
        self.assertEqual(result["logical_ticks"], 0)
        self.assertFalse(result["claimed_success"])

    def test_total_tool_budget_applies_across_model_turns(self):
        responses = [chat_response(tool_calls=[tool_call("wait", {}, f"wait-{turn}-{i}")
                                              for i in range(count)])
                     for turn, count in enumerate((12, 13))]
        result, request = self.run_mocked(responses)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["run_status"], "tool_budget_exhausted")
        self.assertEqual(result["logical_ticks"], 12)
        self.assertEqual(len([m for m in result["messages"] if m["role"] == "tool"]), 12)

    def test_invalid_turn_budget_is_rejected_without_network_request(self):
        for max_turns in (0, -1, 101):
            with self.subTest(max_turns=max_turns):
                with patch("recovery_eval.api_agent.request_chat") as request:
                    with self.assertRaises(ValueError):
                        run_live_case(self.case, "mock-model", self.endpoint, self.secret, max_turns=max_turns)
                    request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
