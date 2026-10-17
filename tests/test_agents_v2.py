"""Model adapter and live recovery loop tests; no external inference calls."""

from copy import deepcopy
import io
import json
import os
import sys
import types
import unittest
from unittest.mock import patch
import urllib.error

from recovery_eval.agents import (
    LocalTransformersAdapter,
    ModelAdapterError,
    ModelResponse,
    OpenAICompatibleAdapter,
    build_messages,
    parse_action,
    run_agent,
)


class SequenceAdapter:
    model_id = "test-model"
    kind = "local_model"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.received = []

    def complete(self, messages):
        self.received.append(deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return ModelResponse(text=response, generated_token_ids=[10, 11], latency_seconds=0.1,
                             usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                             finish_reason="eos")

    def describe(self):
        return {"backend": "unit_test_only", "model_id": self.model_id}


class FakeEnvironment:
    """A tiny independently scored stateful tool double, not a model mock."""

    def __init__(self, case, max_calls=20):
        self.public_task = {"goal": "Create one record", "title": case["title"]}
        self.tool_schemas = [{"type": "function", "function": {
            "name": "create_record", "parameters": {"type": "object"}}}]
        self.effects = []
        self.calls = []
        self.max_calls = max_calls

    def invoke(self, name, arguments):
        self.calls.append((name, deepcopy(arguments)))
        if name == "create_record" and arguments == {"title": self.public_task["title"]}:
            self.effects.append(deepcopy(arguments))
            return {"ok": True, "result": {"record_id": f"record-{len(self.effects)}"}}
        return {"ok": False, "error": {"type": "invalid_arguments", "message": "Bad public tool arguments"}}

    def finish(self, claimed_success, agent_id, agent_kind):
        success = len(self.effects) == 1
        return {"success": success, "claimed_success": claimed_success,
                "false_success": claimed_success and not success,
                "tool_calls": len(self.calls), "ledger": deepcopy(self.effects),
                "agent_id": agent_id, "agent_kind": agent_kind}


class ActionParsingTests(unittest.TestCase):
    def test_strict_tool_and_finish_actions(self):
        self.assertEqual(parse_action('{"tool":"wait","arguments":{}}')["tool"], "wait")
        self.assertEqual(parse_action('{"finish":{"success":false}}'), {"finish": {"success": False}})
        self.assertEqual(parse_action('{"finish":{"success":true,"message":"Done"}}')["finish"]["message"], "Done")

    def test_rejects_extraction_coercion_duplicate_keys_and_extra_fields(self):
        invalid = [
            '```json\n{"finish":{"success":true}}\n```',
            'Explanation: {"finish":{"success":true}}',
            '{"finish":{"success":"true"}}', '{"finish":{"success":1}}',
            '{"finish":{"success":true,"success":false}}',
            '{"tool":"wait","arguments":{},"success":true}',
            '{"tool":"wait","arguments":[]}', '{"tool":"","arguments":{}}',
            '{"tool":"wait","arguments":{"ticks":NaN}}',
            '{"finish":{"success":true}} {"finish":{"success":false}}',
            '[]', 'null', '{"finish":{"success":true,"extra":1}}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action(raw)

    def test_prompt_variants_share_public_contracts(self):
        public = {"goal": "Create one record"}
        schemas = [{"type": "function", "function": {"name": "create_record"}}]
        basic = build_messages(public, schemas, "basic")
        guarded = build_messages(public, schemas, "guarded")
        self.assertEqual(basic[1], guarded[1])
        self.assertIn("operation_id", guarded[0]["content"])
        self.assertNotIn("operation_id", basic[0]["content"])
        self.assertIn("create_record", basic[0]["content"])
        with self.assertRaises(ValueError):
            build_messages(public, schemas, "unknown")


class LiveLoopTests(unittest.TestCase):
    def run_case(self, responses, **kwargs):
        adapter = SequenceAdapter(responses)
        case = {"title": "Investigate a service request", "fault": "HIDDEN_FAULT_CANARY",
                "oracle": "HIDDEN_ORACLE_CANARY"}
        fake_engine = types.ModuleType("recovery_eval.engine")
        fake_engine.Environment = FakeEnvironment
        with patch.dict(sys.modules, {"recovery_eval.engine": fake_engine}):
            result = run_agent(case, adapter, **kwargs)
        return result, adapter

    def test_model_actions_execute_tools_and_independent_scorer(self):
        result, adapter = self.run_case([
            '{"tool":"create_record","arguments":{"title":"Investigate a service request"}}',
            '{"finish":{"success":true}}',
        ])
        self.assertTrue(result["success"])
        self.assertFalse(result["false_success"])
        self.assertEqual(result["actual_tool_invocations"], 1)
        self.assertEqual(result["evaluation_mode"], "live_model_recovery")
        self.assertEqual(result["model_calls"][0]["generated_token_ids"], [10, 11])
        self.assertEqual(result["model_calls"][0]["usage"]["total_tokens"], 12)
        observation = json.loads(adapter.received[1][-1]["content"])
        self.assertTrue(observation["observation"]["response"]["ok"])

    def test_success_claim_does_not_override_oracle(self):
        result, _ = self.run_case(['{"finish":{"success":true}}'])
        self.assertFalse(result["success"])
        self.assertTrue(result["false_success"])

    def test_case_fault_and_oracle_are_never_in_messages(self):
        result, adapter = self.run_case(['{"finish":{"success":false}}'])
        encoded = json.dumps(adapter.received) + json.dumps(result["messages"])
        self.assertNotIn("HIDDEN_FAULT_CANARY", encoded)
        self.assertNotIn("HIDDEN_ORACLE_CANARY", encoded)
        self.assertFalse(result["labels_in_prompt"])

    def test_one_format_correction_is_explicit_and_recorded(self):
        result, adapter = self.run_case(['bad output', '{"finish":{"success":false}}'])
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(result["format_corrections"], 1)
        self.assertEqual(result["model_calls"][0]["raw_generation"], "bad output")
        self.assertTrue(result["model_calls"][1]["format_correction_request"])
        self.assertIn("FORMAT ERROR", adapter.received[1][-1]["content"])
        self.assertTrue(all(set(message) == {"role", "content"} for message in result["messages"]))

    def test_second_malformed_response_stops_without_silent_repair(self):
        result, _ = self.run_case(['bad', '{"finish":{"success":"true"}}'])
        self.assertEqual(result["run_status"], "invalid_action")
        self.assertEqual(result["model_call_count"], 2)
        self.assertFalse(result["silent_output_repair"])
        self.assertFalse(result["claimed_success"])

    def test_format_retry_can_be_disabled(self):
        result, _ = self.run_case(['bad'], format_retries=0)
        self.assertEqual(result["run_status"], "invalid_action")
        self.assertEqual(result["format_corrections"], 0)

    def test_format_correction_counts_toward_turn_budget(self):
        result, adapter = self.run_case(['bad'], max_turns=1)
        self.assertEqual(result["run_status"], "turn_budget_exhausted")
        self.assertEqual(len(adapter.received), 1)
        self.assertFalse(result["claimed_success"])

    def test_tool_budget_prevents_extra_effect(self):
        action = '{"tool":"create_record","arguments":{"title":"Investigate a service request"}}'
        result, _ = self.run_case([action, action], max_tool_calls=1)
        self.assertEqual(result["run_status"], "tool_budget_exhausted")
        self.assertEqual(len(result["ledger"]), 1)
        self.assertEqual(result["actual_tool_invocations"], 1)

    def test_invalid_tool_arguments_are_observed_and_do_not_write(self):
        result, adapter = self.run_case([
            '{"tool":"create_record","arguments":{}}', '{"finish":{"success":false}}'
        ])
        self.assertEqual(result["ledger"], [])
        self.assertFalse(json.loads(adapter.received[1][-1]["content"])["observation"]["response"]["ok"])

    def test_adapter_error_records_type_without_secret_exception_text(self):
        result, _ = self.run_case([RuntimeError("SECRET_TEST_KEY")])
        self.assertEqual(result["run_status"], "infrastructure_error")
        self.assertEqual(result["model_calls"][0]["error_type"], "RuntimeError")
        self.assertNotIn("SECRET_TEST_KEY", json.dumps(result))

    def test_budgets_require_bounded_integers(self):
        for kwargs in ({"max_turns": True}, {"max_tool_calls": 0}, {"format_retries": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_case([], **kwargs)


class BackendTests(unittest.TestCase):
    def test_local_initialization_does_not_load_weights(self):
        adapter = LocalTransformersAdapter(local_files_only=True)
        self.assertIsNone(adapter._model)
        self.assertTrue(adapter.describe()["local_files_only"])
        self.assertFalse(adapter.describe()["do_sample"])

    def test_api_key_absent_sends_no_request(self):
        adapter = OpenAICompatibleAdapter("test", "https://endpoint.example.test/v1", api_key_env="MISSING_TEST_KEY")
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen") as request:
            with self.assertRaises(ModelAdapterError):
                adapter.complete([{"role": "user", "content": "Hi"}])
        request.assert_not_called()

    def test_api_uses_text_protocol_and_never_logs_key(self):
        adapter = OpenAICompatibleAdapter("test", "https://endpoint.example.test/v1", api_key_env="TEST_MODEL_KEY")
        provider_response = {"choices": [{"message": {"content": '{"finish":{"success":false}}'},
                                           "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}}
        with patch.dict(os.environ, {"TEST_MODEL_KEY": "SECRET_TEST_KEY"}), patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(provider_response).encode())
        ) as request:
            result = adapter.complete([{"role": "user", "content": "Hi"}])
        payload = json.loads(request.call_args.args[0].data)
        self.assertNotIn("tools", payload)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(result.usage["total_tokens"], 21)
        self.assertIsNone(result.generated_token_ids)
        self.assertNotIn("SECRET_TEST_KEY", json.dumps(adapter.describe()) + json.dumps(result.__dict__))

    def test_api_endpoint_validation_prevents_embedded_credentials(self):
        for url in ("https://user:secret@example.test/v1", "https://example.test/v1?key=secret",
                    "http://remote.example.test/v1", "file:///tmp/test"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OpenAICompatibleAdapter("test", url)

    def test_api_http_errors_retain_code_only(self):
        adapter = OpenAICompatibleAdapter("test", "https://endpoint.example.test/v1", api_key_env="TEST_MODEL_KEY")
        failure = urllib.error.HTTPError("https://example.test/SECRET_TEST_KEY", 401,
                                         "SECRET_TEST_KEY", {}, None)
        with patch.dict(os.environ, {"TEST_MODEL_KEY": "SECRET_TEST_KEY"}), patch(
            "urllib.request.urlopen", side_effect=failure
        ), self.assertRaises(ModelAdapterError) as caught:
            adapter.complete([{"role": "user", "content": "Hi"}])
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("SECRET_TEST_KEY", str(caught.exception))

    def test_api_rejects_nontext_response(self):
        adapter = OpenAICompatibleAdapter("test", "https://endpoint.example.test/v1", api_key_env="TEST_MODEL_KEY")
        body = {"choices": [{"message": {"content": None, "tool_calls": []}}]}
        with patch.dict(os.environ, {"TEST_MODEL_KEY": "SECRET_TEST_KEY"}), patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())
        ), self.assertRaises(ModelAdapterError):
            adapter.complete([{"role": "user", "content": "Hi"}])


class EngineIntegrationTests(unittest.TestCase):
    def test_clean_plan_is_executed_by_text_actions_in_real_environment(self):
        from recovery_eval.engine import Environment
        from recovery_eval.workflows import build_matrix

        case = build_matrix(workflows="ticket_dispatch", variants=1, seeds=1, faults="clean")[0]
        public = Environment(case).public_task
        actions = [json.dumps({"tool": step["tool"], "arguments": step["arguments"]})
                   for step in public["steps"]]
        result = run_agent(case, SequenceAdapter(actions + ['{"finish":{"success":true}}']))
        self.assertTrue(result["success"])
        self.assertEqual(len(result["ledger"]), 4)
        self.assertEqual(result["actual_tool_invocations"], 4)
        self.assertEqual(result["run_status"], "completed")

    def test_lost_ack_retry_with_stable_key_uses_real_deduplication(self):
        from recovery_eval.engine import Environment
        from recovery_eval.workflows import build_matrix

        case = build_matrix(workflows="ticket_dispatch", variants=1, seeds=1, faults="lost_ack")[0]
        steps = Environment(case).public_task["steps"]
        actions = []
        for index, step in enumerate(steps):
            arguments = {**step["arguments"], "operation_id": f"operation-{step['name']}"}
            action = json.dumps({"tool": step["tool"], "arguments": arguments})
            actions.append(action)
            if index == 0:
                actions.append(action)
        result = run_agent(case, SequenceAdapter(actions + ['{"finish":{"success":true}}']),
                           prompt_variant="guarded")
        self.assertFalse(result["model_calls"][0]["tool_response"]["ok"])
        self.assertTrue(result["model_calls"][1]["tool_response"]["ok"])
        self.assertTrue(result["success"])
        self.assertEqual(result["duplicates"], 0)
        self.assertEqual(len(result["ledger"]), 4)


if __name__ == "__main__":
    unittest.main()
