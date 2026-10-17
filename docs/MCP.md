# MCP stdio integration

The MCP adapter exposes one **simulated** recovery episode to an external agent
using the real MCP Python SDK and a stdio subprocess. It does not send email,
charge money, book rooms, publish documents, open arbitrary paths, or execute
shell commands. All business effects exist only inside the simulator.

## Install and launch

From the repository root, with Python 3.11 or later:

```bash
python -m pip install -e .
python -m pip install -r requirements-mcp.txt
python -m recovery_eval.mcp_server --workflow ticket_dispatch --result-out results/mcp/episode.json
```

The server waits for MCP JSON-RPC on stdin. A lack of terminal output is normal.
Do not print additional text to its stdout. Connect it through an MCP-capable
client or use the integration tests below.

The dependency is pinned to `mcp==2.0.0`. This version uses
`from mcp.server import MCPServer`; it does not have the historical
`mcp.server.fastmcp.FastMCP` module. The tested high-level registration API is
`@server.tool()` and `server.run(transport="stdio")`.

Available clean demo workflows are:

1. `ticket_dispatch`
2. `inventory_order`
3. `meeting_booking`
4. `document_publication`

## MCP client configuration

Set `cwd` to the local checkout and use the Python executable where the optional
dependency is installed. For clients that support a JSON `mcpServers` config:

```json
{
  "mcpServers": {
    "recovery-eval": {
      "command": "python",
      "args": [
        "-m", "recovery_eval.mcp_server",
        "--workflow", "ticket_dispatch",
        "--result-out", "results/mcp/episode.json"
      ],
      "cwd": "/absolute/path/to/agent_recovery_eval"
    }
  }
}
```

This is configuration for an external client, not an assertion that a particular
desktop application's configuration has been installed or tested. One server
process holds one episode. Start a fresh process for each test case; continuing
to use an already finished process does not reset the episode.

## Public tools

| Tool | Arguments | Return | Boundary |
|---|---|---|---|
| `get_task` | None | `task` and dynamic `tools` schemas | Goal, provided step plan, public service contract and call budget |
| `execute` | `tool_name: str`, `arguments: object` | Public success result or public tool error | Only registered simulated business operations and `inspect_step` |
| `finish` | `claimed_success: bool` | Exactly `{"closed": true}` | Records the client's claim and closes the episode without returning a score |

`get_task.tools` is a list of OpenAI-style function schemas. Each supplied step
also includes its resolved `arguments` and `requires` prerequisites. To invoke a
business operation, call MCP `execute` with that operation's name and argument
object. For example, using values obtained from the public task:

```json
{
  "tool_name": "create_ticket",
  "arguments": {
    "ticket_ref": "T-001",
    "title": "Restore service 1",
    "severity": "low",
    "operation_id": "episode-1:create"
  }
}
```

The stable `operation_id` is optional when supported by the service contract.
A timeout has an unknown outcome. Repeating a write without a stable key can
produce duplicate effects; a stale `inspect_step` result does not cancel an
already pending write. Agents must reason from these public observations.

After `finish`, `get_task` returns `{"closed": true}` and `execute` returns a
public `episode_closed` error. Calling `finish` again does not change the first
claim or saved result. Public tool calls are serialized to preserve episode
ordering, even when an MCP client submits concurrent requests.

## Evaluator-owned cases and result files

The evaluator may create a case file and pass it through the launch arguments:

```bash
python -c "import json; from pathlib import Path; from recovery_eval.workflows import build_matrix; p=Path('results/mcp'); p.mkdir(parents=True, exist_ok=True); (p/'case.json').write_text(json.dumps(build_matrix('ticket_dispatch', variants=1, seeds=1, faults='lost_ack')[0]), encoding='utf-8')"
python -m recovery_eval.mcp_server --case-json results/mcp/case.json --result-out results/mcp/episode.json
```

`--case-json` is a single evaluator-owned JSON object, not a JSONL file.
`--workflow` selects a clean demo and cannot be combined with `--case-json`.
Other evaluator controls are `--max-calls` (1–1000, default 24),
`--no-idempotency`, and `--agent-id`.

Only the evaluator should read the case file and result file. MCP tools expose
neither paths nor file operations. The full result contains the hidden condition,
independent oracle verdict, trace, committed-effect ledger and final state. It is
written atomically at `finish`; on normal stdin disconnect before `finish`, the
server saves an incomplete episode with `claimed_success=false` and
`close_reason="transport_closed_without_finish"`. A force-killed process cannot
guarantee final persistence. Omitting `--result-out` discards the private result.

This stdio boundary prevents ordinary MCP responses from disclosing oracle
internals. It is **not an OS sandbox**: an agent that also has arbitrary local
filesystem or process access could bypass this boundary. Give an external agent
only the intended MCP tools, or isolate its OS access independently. No claim of
adversarial in-process Python isolation is made.

## Actual protocol integration tests

```bash
python -m unittest discover -s tests -p test_mcp_v2.py -v
```

These tests use `mcp.ClientSession` and `mcp.client.stdio.stdio_client` to spawn
`python -m recovery_eval.mcp_server`, perform protocol initialization, list tools,
and call `get_task`, `execute` and `finish`. They cover:

1. Clean execution of all four supplied workflows.
2. Lost acknowledgements and stable-key retries.
3. A duplicate-producing failure whose oracle score remains in the private file.
4. Normal disconnect persistence with an honest incomplete claim.
5. Unknown-tool rejection, closed-episode rejection and repeated-finish stability.
6. Business-tool call-budget enforcement.

The suite skips these optional tests when MCP 2.0 is not installed. No test uses
a mocked MCP transport, model API, Modal account, Prime Intellect account, or
external business service. Passing these tests verifies the local MCP transport
integration, not a language model's recovery ability.
