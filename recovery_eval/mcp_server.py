"""Run one isolated recovery episode as an MCP 2.0 stdio server.

Only public task information and ordinary business-tool observations cross the
MCP boundary. Oracle scores and configuration are written exclusively to a
local evaluator-owned result file after the client closes the episode.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any


class EpisodeSession:
    """Serialize client actions and keep evaluator results out of tool returns."""

    def __init__(self, environment: Any, result_out: str | Path | None = None,
                 agent_id: str = "mcp-client") -> None:
        self.environment = environment
        self.result_out = Path(result_out) if result_out is not None else None
        self.agent_id = agent_id
        self._lock = threading.RLock()
        self._result: dict[str, Any] | None = None

    def get_task(self) -> dict[str, Any]:
        with self._lock:
            if self._result is not None:
                return {"closed": True}
            return {"task": deepcopy(self.environment.public_task),
                    "tools": deepcopy(self.environment.tool_schemas)}

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._result is not None:
                return {"ok": False, "error": {"type": "episode_closed",
                        "message": "The episode is closed.", "retryable": False}}
            # Environment.invoke only dispatches its declared business tools.
            return self.environment.invoke(tool_name, arguments)

    def finish(self, claimed_success: bool, *, close_reason: str = "client_finish") -> dict[str, bool]:
        with self._lock:
            if type(claimed_success) is not bool:
                raise ValueError("claimed_success must be a boolean")
            if self._result is None:
                self._result = self.environment.finish(
                    claimed_success, agent_id=self.agent_id, agent_kind="mcp_client",
                    transport="mcp_stdio", close_reason=close_reason)
                self._persist_result()
            # Scores, ledger, labels, hashes and hidden conditions stay private.
            return {"closed": True}

    def close(self) -> None:
        """Save an honestly incomplete episode on normal stdin disconnect."""
        with self._lock:
            if self._result is None:
                self.finish(False, close_reason="transport_closed_without_finish")

    def _persist_result(self) -> None:
        if self.result_out is None:
            return
        self.result_out.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".mcp-result-", suffix=".json", dir=self.result_out.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self._result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.result_out)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def create_server(session: EpisodeSession) -> Any:
    """Build the SDK server lazily so the base package needs no MCP install."""
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError("Install the optional MCP SDK: pip install -r requirements-mcp.txt") from exc

    server = MCPServer(
        "Recovery Eval Episode", version="2.0.0", log_level="WARNING",
        instructions=("Call get_task, execute the declared business tools using execute, "
                      "and call finish with an honest claimed_success boolean. "
                      "The server simulates one business workflow. It does not provide "
                      "oracle scores or hidden evaluation conditions."))

    @server.tool()
    def get_task() -> dict[str, Any]:
        """Get the public goal, supplied step plan, service contract and tool schemas."""
        return session.get_task()

    @server.tool()
    def execute(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one declared simulated business tool; outcome can be unknown on timeout."""
        return session.execute(tool_name, arguments)

    @server.tool()
    def finish(claimed_success: bool) -> dict[str, bool]:
        """Close this episode with your completion claim. No score is returned."""
        return session.finish(claimed_success)

    return server


def main(argv: list[str] | None = None) -> None:
    from .engine import Environment
    from .workflows import build_matrix, workflow_specs

    parser = argparse.ArgumentParser(description="Serve one simulated recovery episode over MCP stdio.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-json", type=Path, help="Evaluator-owned JSON object for one case.")
    selection.add_argument("--workflow", choices=tuple(workflow_specs()), default="ticket_dispatch",
                           help="Use a clean demo case; defaults to ticket_dispatch.")
    parser.add_argument("--result-out", type=Path, help="Private evaluator result JSON, never an MCP response.")
    parser.add_argument("--max-calls", type=int, default=24, help="Business-tool call budget, 1 to 1000.")
    parser.add_argument("--agent-id", default="mcp-client", help="Evaluator-side result identifier.")
    parser.add_argument("--no-idempotency", action="store_true", help="Disable the simulated key contract.")
    args = parser.parse_args(argv)
    if not 1 <= args.max_calls <= 1000:
        parser.error("--max-calls must be between 1 and 1000")
    try:
        if args.case_json is not None:
            with args.case_json.open(encoding="utf-8") as stream:
                case = json.load(stream)
            if not isinstance(case, dict):
                raise ValueError("one case object is required")
        else:
            case = build_matrix(args.workflow, variants=1, seeds=1, faults="clean")[0]
        environment = Environment(case, max_calls=args.max_calls,
                                  idempotency=False if args.no_idempotency else None)
        session = EpisodeSession(environment, args.result_out, args.agent_id)
        server = create_server(session)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        # Avoid echoing evaluator-owned case contents or filesystem paths.
        parser.error(f"Unable to initialize episode ({type(exc).__name__}). Check the case and optional SDK.")
    try:
        server.run(transport="stdio")
    finally:
        session.close()


if __name__ == "__main__":
    main()
