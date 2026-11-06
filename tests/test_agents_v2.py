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
    native_tools,
    parse_action,
    parse_native_action,
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


class NativeSequenceAdapter:
    """Returns provider-style tool calls; a list item is one turn's calls, a str is prose."""

    model_id = "test-model"
    kind = "api_model"
    supports_native_tools = True

    def __init__(self, turns):
        self.turns = iter(turns)
        self.received = []

    def complete(self, messages, tools=None):
        self.received.append((deepcopy(messages), deepcopy(tools)))
        turn = next(self.turns)
        if isinstance(turn, str):
            return ModelResponse(text=turn, tool_calls=[])
        calls = [{"id": f"call-{len(self.received)}-{i}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(arguments)}}
                 for i, (name, arguments) in enumerate(turn)]
        return ModelResponse(text="", tool_calls=calls)

    def describe(self):
        return {"backend": "unit_test_only", "model_id": self.model_id}


def native_turns_for(case, *, repeat_first=False, keyed=False):
    from recovery_eval.engine import Environment

    turns = []
    for index, step in enumerate(Environment(case).public_task["steps"]):
        arguments = dict(step["arguments"])
        if keyed:
            arguments["operation_id"] = f"operation-{step['name']}"
        turns.append([(step["tool"], arguments)])
        if repeat_first and index == 0:
            turns.append([(step["tool"], arguments)])
    return turns + [[("finish", {"success": True})]]


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


class NativeProtocolTests(unittest.TestCase):
    def test_parse_native_action_is_as_strict_as_text_actions(self):
        call = lambda name, args: [{"id": "c", "type": "function", "function": {"name": name, "arguments": args}}]
        self.assertEqual(parse_native_action(call("create", '{"a":1}')), {"tool": "create", "arguments": {"a": 1}})
        self.assertEqual(parse_native_action(call("finish", '{"success":false}')), {"finish": {"success": False}})
        for bad in (None, [], call("create", '{"a":1,"a":2}'), call("create", "[1]"),
                    call("finish", '{"success":"yes"}'), call("create", '{"a":NaN}'),
                    call("create", '{"a":1}') + call("create", '{"a":1}')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_native_action(bad)

    def test_native_tools_add_finish_and_keep_public_contracts(self):
        schemas = [{"type": "function", "function": {"name": "create", "parameters": {"type": "object"}}}]
        tools = native_tools(schemas)
        self.assertEqual(tools[0], schemas[0])
        self.assertEqual(tools[-1]["function"]["name"], "finish")
        with self.assertRaises(ValueError):
            native_tools([{"type": "function", "function": {"name": "finish"}}])

    def test_native_messages_keep_the_same_advice_but_move_contracts_to_tools(self):
        task = {"task_id": "t"}
        schemas = [{"type": "function", "function": {"name": "create_record"}}]
        native = build_messages(task, schemas, "guarded", "native")
        text = build_messages(task, schemas, "guarded")
        self.assertNotIn("create_record", native[0]["content"])
        self.assertIn("create_record", text[0]["content"])
        self.assertIn("stable operation_id", native[0]["content"])
        self.assertEqual(native[1], text[1])

    def test_native_episode_executes_tools_with_tool_messages(self):
        from recovery_eval.workflows import build_matrix

        case = build_matrix(workflows="ticket_dispatch", variants=1, seeds=1, faults="clean")[0]
        adapter = NativeSequenceAdapter(native_turns_for(case))
        result = run_agent(case, adapter, tool_protocol="native")
        self.assertTrue(result["success"])
        self.assertEqual(result["tool_protocol"], "native")
        self.assertTrue(result["agent_id"].endswith(":basic:native"))
        _, tools = adapter.received[0]
        self.assertEqual(tools[-1]["function"]["name"], "finish")
        messages, _ = adapter.received[1]
        self.assertEqual(messages[-2]["tool_calls"][0]["id"], messages[-1]["tool_call_id"])
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertTrue(json.loads(messages[-1]["content"])["observation"]["response"]["ok"])

    def test_native_lost_ack_retry_with_stable_key(self):
        from recovery_eval.decisions import retry_audit
        from recovery_eval.workflows import build_matrix

        case = build_matrix(workflows="ticket_dispatch", variants=1, seeds=1, faults="lost_ack")[0]
        result = run_agent(case, NativeSequenceAdapter(native_turns_for(case, repeat_first=True, keyed=True)),
                           prompt_variant="guarded", tool_protocol="native")
        self.assertTrue(result["success"])
        self.assertEqual(result["duplicates"], 0)
        self.assertEqual(retry_audit(result)["justified_retries"], 1)

    def test_several_calls_in_one_turn_execute_nothing_and_answer_every_id(self):
        from recovery_eval.workflows import build_matrix

        case = build_matrix(workflows="ticket_dispatch", variants=1, seeds=1, faults="clean")[0]
        turns = native_turns_for(case)
        adapter = NativeSequenceAdapter([turns[0] + turns[1], "I am done."])
        result = run_agent(case, adapter, tool_protocol="native")
        self.assertEqual(result["actual_tool_invocations"], 0)
        self.assertEqual(result["ledger"], [])
        self.assertEqual(result["run_status"], "invalid_action")
        self.assertEqual(result["format_corrections"], 1)
        messages, _ = adapter.received[1]
        answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
        self.assertEqual(answered, [c["id"] for c in messages[-3]["tool_calls"]])
        self.assertIn("No tool was executed", messages[-1]["content"])

    def test_native_protocol_requires_a_supporting_backend(self):
        with self.assertRaises(ValueError):
            run_agent({"title": "x"}, SequenceAdapter([]), tool_protocol="native")

    def test_api_sends_tools_and_reads_tool_calls(self):
        adapter = OpenAICompatibleAdapter("test", "https://endpoint.example.test/v1", api_key_env="TEST_MODEL_KEY")
        provider_response = {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "index": 0,
             "function": {"name": "finish", "arguments": '{"success":true}'}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
        tools = native_tools([])
        with patch.dict(os.environ, {"TEST_MODEL_KEY": "SECRET_TEST_KEY"}), patch(
            "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(provider_response).encode())
        ) as request:
            result = adapter.complete([{"role": "user", "content": "Hi"}], tools=tools)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(result.text, "")
        self.assertEqual(result.tool_calls, [{"id": "call_1", "type": "function",
                                              "function": {"name": "finish", "arguments": '{"success":true}'}}])
        self.assertEqual(parse_native_action(result.tool_calls), {"finish": {"success": True}})


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
