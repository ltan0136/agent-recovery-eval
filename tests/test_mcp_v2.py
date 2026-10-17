"""Exercise the actual MCP SDK over stdio subprocesses, without model services."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from recovery_eval.workflows import build_matrix, workflow_specs

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.server import MCPServer  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_KEYS = {"fault", "target_step", "case_config", "execution_config", "ledger",
                "trace", "final_state", "failure_reasons", "case_hash", "config_hash"}


def payload(result):
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    text = next(item.text for item in result.content if item.type == "text")
    return json.loads(text)


def all_keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(all_keys(item) for item in value), set())
    return set()


@unittest.skipUnless(MCP_AVAILABLE, "Install requirements-mcp.txt to run real MCP integration tests")
class MCPStdioIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="recovery-mcp-test-")
        self.addCleanup(self.directory.cleanup)
        self.result_out = Path(self.directory.name) / "private-result.json"
        self.stderr = open(Path(self.directory.name) / "server.stderr", "w+", encoding="utf-8")
        self.addCleanup(self.stderr.close)

    @asynccontextmanager
    async def client(self, *, case=None, workflow=None, extra_args=()):
        args = ["-m", "recovery_eval.mcp_server", "--result-out", str(self.result_out)]
        if case is not None:
            case_path = Path(self.directory.name) / "private-case.json"
            case_path.write_text(json.dumps(case), encoding="utf-8")
            args.extend(["--case-json", str(case_path)])
        elif workflow is not None:
            args.extend(["--workflow", workflow])
        args.extend(extra_args)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), environment.get("PYTHONPATH", "")])
        parameters = StdioServerParameters(command=sys.executable, args=args,
                                           cwd=str(ROOT), env=environment)
        async with stdio_client(parameters, errlog=self.stderr) as streams:
            async with ClientSession(*streams, read_timeout_seconds=15) as session:
                initialized = await session.initialize()
                self.assertEqual(initialized.server_info.name, "Recovery Eval Episode")
                yield session

    async def call(self, session, name, arguments=None):
        result = await session.call_tool(name, arguments or {})
        self.assertFalse(result.is_error, result)
        data = payload(result)
        self.assertFalse(PRIVATE_KEYS & all_keys(data), data)
        return data

    def read_result(self):
        return json.loads(self.result_out.read_text(encoding="utf-8"))

    async def test_clean_workflows_run_over_real_stdio(self):
        for workflow in workflow_specs():
            with self.subTest(workflow=workflow):
                async with self.client(workflow=workflow) as session:
                    tools = await session.list_tools()
                    self.assertEqual({tool.name for tool in tools.tools}, {"get_task", "execute", "finish"})
                    public = await self.call(session, "get_task")
                    self.assertEqual(public["task"]["workflow_id"], workflow)
                    declared = {item["function"]["name"] for item in public["tools"]}
                    self.assertIn("inspect_step", declared)
                    for step in public["task"]["steps"]:
                        self.assertIn(step["tool"], declared)
                        arguments = dict(step["arguments"], operation_id=f"mcp:{step['name']}")
                        response = await self.call(session, "execute", {
                            "tool_name": step["tool"], "arguments": arguments})
                        self.assertTrue(response["ok"], response)
                    self.assertEqual(await self.call(session, "finish", {"claimed_success": True}), {"closed": True})
                row = self.read_result()
                self.assertTrue(row["success"])
                self.assertEqual(row["agent_kind"], "mcp_client")
                self.assertEqual(row["transport"], "mcp_stdio")
                self.assertEqual(row["close_reason"], "client_finish")

    async def test_lost_ack_retry_is_observable_without_oracle_disclosure(self):
        case = build_matrix("ticket_dispatch", variants=1, seeds=1, faults="lost_ack")[0]
        async with self.client(case=case) as session:
            public = await self.call(session, "get_task")
            self.assertNotIn("lost_ack", json.dumps(public))
            for index, step in enumerate(public["task"]["steps"]):
                args = dict(step["arguments"], operation_id=f"stable:{step['name']}")
                response = await self.call(session, "execute", {"tool_name": step["tool"], "arguments": args})
                if index == 0:
                    self.assertFalse(response["ok"])
                    self.assertEqual(response["error"]["type"], "timeout")
                    response = await self.call(session, "execute", {"tool_name": step["tool"], "arguments": args})
                self.assertTrue(response["ok"], response)
            self.assertEqual(await self.call(session, "finish", {"claimed_success": True}), {"closed": True})
        row = self.read_result()
        self.assertEqual(row["fault"], "lost_ack")
        self.assertTrue(row["success"])
        self.assertEqual(row["duplicates"], 0)
        self.assertTrue(row["trace"])
        self.assertTrue(row["ledger"])

    async def test_failed_episode_score_stays_in_private_file(self):
        case = build_matrix("ticket_dispatch", variants=1, seeds=1, faults="lost_ack")[0]
        async with self.client(case=case, extra_args=("--no-idempotency",)) as session:
            public = await self.call(session, "get_task")
            for index, step in enumerate(public["task"]["steps"]):
                args = {"tool_name": step["tool"], "arguments": step["arguments"]}
                response = await self.call(session, "execute", args)
                if index == 0:
                    self.assertFalse(response["ok"])
                    response = await self.call(session, "execute", args)
                self.assertTrue(response["ok"])
            self.assertEqual(await self.call(session, "finish", {"claimed_success": True}), {"closed": True})
        row = self.read_result()
        self.assertFalse(row["success"])
        self.assertTrue(row["false_success"])
        self.assertGreater(row["duplicates"], 0)

    async def test_normal_disconnect_saves_honest_incomplete_result(self):
        async with self.client() as session:
            await self.call(session, "get_task")
        row = self.read_result()
        self.assertFalse(row["claimed_success"])
        self.assertFalse(row["success"])
        self.assertEqual(row["close_reason"], "transport_closed_without_finish")
        self.assertGreater(row["missing"], 0)

    async def test_unknown_tool_and_closed_episode_are_bounded(self):
        async with self.client() as session:
            response = await self.call(session, "execute", {"tool_name": "grade", "arguments": {}})
            self.assertFalse(response["ok"])
            self.assertEqual(response["error"]["type"], "unknown_tool")
            self.assertEqual(await self.call(session, "finish", {"claimed_success": False}), {"closed": True})
            first = self.read_result()
            response = await self.call(session, "execute", {"tool_name": "inspect_step", "arguments": {"step": "create"}})
            self.assertEqual(response["error"]["type"], "episode_closed")
            self.assertEqual(await self.call(session, "get_task"), {"closed": True})
            self.assertEqual(await self.call(session, "finish", {"claimed_success": True}), {"closed": True})
            self.assertEqual(first, self.read_result())
        self.assertFalse(self.read_result()["claimed_success"])

    async def test_tool_call_budget_is_enforced_in_subprocess(self):
        async with self.client(extra_args=("--max-calls", "1")) as session:
            public = await self.call(session, "get_task")
            self.assertEqual(public["task"]["tool_call_budget"], 1)
            step = public["task"]["steps"][0]
            response = await self.call(session, "execute", {"tool_name": step["tool"], "arguments": step["arguments"]})
            self.assertTrue(response["ok"])
            response = await self.call(session, "execute", {"tool_name": "inspect_step", "arguments": {"step": step["name"]}})
            self.assertEqual(response["error"]["type"], "budget_exhausted")
            await self.call(session, "finish", {"claimed_success": False})
        self.assertEqual(self.read_result()["tool_calls"], 1)


if __name__ == "__main__":
    unittest.main()
