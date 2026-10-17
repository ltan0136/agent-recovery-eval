"""A small deterministic simulator for tool-using recovery policies.

All policies are hand-written baselines, not language-model agents. Every policy
receives the same tool contract and can observe only tool returns or timeouts.
The scorer independently inspects the committed-effect ledger after settling
all scheduled writes. Logical ticks are simulation units, not wall-clock time.
"""

from __future__ import annotations

from copy import deepcopy
import heapq
import hashlib
import json
import random
from typing import Any


FAULTS = (
    "clean",
    "before_commit_once",
    "ack_lost",
    "late_commit",
    "stale_read",
    "duplicate_delivery",
)
POLICIES = ("bounded_retry", "verify_then_retry", "idempotent_retry")
MAX_ATTEMPTS = 4
MAX_POLICY_TICKS = 32


def _condition_hash(case: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ToolTimeout(TimeoutError):
    """The caller does not know whether a write took effect."""


def build_cases(task_count: int = 12, seeds: int = 5) -> list[dict[str, Any]]:
    """Build a factorial task/seed/fault suite with reproducible schedules."""
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        raise ValueError("task_count must be a positive integer")
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise ValueError("seeds must be a positive integer")
    cases = []
    for task_index in range(task_count):
        task_id = f"task-{task_index + 1:03d}"
        payload = {
            "task_id": task_id,
            "title": f"Service request {task_index + 1}",
            "body": f"Investigate issue {task_index + 1} and retain its audit record.",
            "recipient": f"owner-{task_index + 1}@example.test",
            "message": f"Service request {task_index + 1} has been created.",
        }
        for seed in range(seeds):
            # A dedicated PRNG avoids dependence on process hash randomization.
            rng = random.Random(f"{task_id}:{seed}")
            target_op = rng.choice(("create_record", "send_notification"))
            delay = rng.randint(4, 6)
            for fault in FAULTS:
                cases.append({
                    "case_id": f"{task_id}-s{seed}-{fault}",
                    "task_id": task_id,
                    "seed": seed,
                    "fault": fault,
                    "target_op": target_op,
                    "late_commit_delay": delay,
                    "payload": deepcopy(payload),
                })
    return cases


class _ToolServer:
    """One shared server contract, including optional idempotency keys."""

    def __init__(self, case: dict[str, Any]):
        self.case = deepcopy(case)
        self.tick = 0
        self.trace: list[dict[str, Any]] = []
        self.ledger: list[dict[str, Any]] = []
        self.tool_calls = 0
        self._fault_used = False
        self._stale_read_pending = False
        self._schedule: list[tuple[int, int, str, dict[str, Any], str | None]] = []
        self._scheduled_index = 0
        self._requests: dict[tuple[str, str], dict[str, Any]] = {}
        self._record_count = 0
        self._notification_count = 0

    def _event(self, event: str, **fields: Any) -> None:
        self.trace.append({"index": len(self.trace), "tick": self.tick, "event": event, **deepcopy(fields)})

    def _advance(self, amount: int = 1) -> None:
        target_tick = self.tick + amount
        while self._schedule and self._schedule[0][0] <= target_tick:
            due, _, operation, payload, operation_id = heapq.heappop(self._schedule)
            self.tick = due
            self._event("scheduled_delivery", operation=operation, operation_id=operation_id)
            self._commit(operation, payload, operation_id, source="late_delivery")
        self.tick = target_tick

    def wait(self, ticks: int = 1) -> None:
        if self.tick + ticks > MAX_POLICY_TICKS:
            raise RuntimeError("policy logical-tick budget exhausted")
        self._event("policy_wait", ticks=ticks)
        self._advance(ticks)

    def _commit(self, operation: str, payload: dict[str, Any], operation_id: str | None,
                source: str = "tool_call") -> dict[str, Any]:
        key = (operation, operation_id) if operation_id is not None else None
        if key is not None:
            request = self._requests.get(key)
            if request and request["payload"] != payload:
                raise ValueError("operation_id reused with a different payload")
            if request and request["state"] == "committed":
                self._event("delivery_deduplicated", operation=operation,
                            operation_id=operation_id, source=source)
                return deepcopy(request["result"])
        if operation == "create_record":
            self._record_count += 1
            result = {"record_id": f"record-{self._record_count:03d}"}
        else:
            self._notification_count += 1
            result = {"notification_id": f"notification-{self._notification_count:03d}",
                      "record_id": payload["record_id"]}
        effect = {
            "effect_id": f"effect-{len(self.ledger) + 1:03d}",
            "tick": self.tick,
            "operation": operation,
            "operation_id": operation_id,
            "payload": deepcopy(payload),
            "result": deepcopy(result),
            "source": source,
        }
        self.ledger.append(effect)
        if key is not None:
            self._requests[key] = {"state": "committed", "payload": deepcopy(payload), "result": deepcopy(result)}
        self._event("effect_committed", **effect)
        return result

    def _timeout(self, operation: str) -> None:
        # The public error intentionally does not reveal commit state.
        self._event("tool_timeout", operation=operation, error="request timed out; outcome unknown")
        raise ToolTimeout("request timed out; outcome unknown")

    def mutate(self, operation: str, payload: dict[str, Any],
               operation_id: str | None = None) -> dict[str, Any]:
        if operation not in ("create_record", "send_notification"):
            raise ValueError(f"unknown mutating operation: {operation}")
        if self.tick >= MAX_POLICY_TICKS:
            raise RuntimeError("policy logical-tick budget exhausted")
        self._advance()
        self.tool_calls += 1
        self._event("tool_call", operation=operation, payload=payload, operation_id=operation_id)
        key = (operation, operation_id) if operation_id is not None else None
        if key is not None and key in self._requests:
            request = self._requests[key]
            if request["payload"] != payload:
                self._event("tool_error", operation=operation, error="idempotency key payload conflict")
                raise ValueError("operation_id reused with a different payload")
            if request["state"] == "pending":
                self._timeout(operation)
            self._event("request_deduplicated", operation=operation, operation_id=operation_id)
            result = deepcopy(request["result"])
            self._event("tool_return", operation=operation, result=result)
            return result
        apply_fault = not self._fault_used and operation == self.case["target_op"]
        fault = self.case["fault"] if apply_fault else "clean"
        if apply_fault:
            self._fault_used = True
            self._event("fault_triggered", operation=operation, fault=fault)
        if fault == "before_commit_once":
            self._timeout(operation)
        if fault == "late_commit":
            due = self.tick + self.case["late_commit_delay"]
            self._scheduled_index += 1
            heapq.heappush(self._schedule, (due, self._scheduled_index, operation,
                                           deepcopy(payload), operation_id))
            if key is not None:
                self._requests[key] = {"state": "pending", "payload": deepcopy(payload)}
            self._event("write_scheduled", operation=operation, due_tick=due, operation_id=operation_id)
            self._timeout(operation)
        result = self._commit(operation, payload, operation_id)
        if fault in ("ack_lost", "stale_read"):
            if fault == "stale_read":
                self._stale_read_pending = True
            self._timeout(operation)
        if fault == "duplicate_delivery":
            self._event("duplicate_delivery", operation=operation, operation_id=operation_id)
            self._commit(operation, payload, operation_id, source="duplicate_delivery")
        self._event("tool_return", operation=operation, result=result)
        return deepcopy(result)

    def read_status(self, operation: str, expected_payload: dict[str, Any]) -> dict[str, Any]:
        if self.tick >= MAX_POLICY_TICKS:
            raise RuntimeError("policy logical-tick budget exhausted")
        self._advance()
        self.tool_calls += 1
        self._event("tool_call", operation="read_status", target_operation=operation, query=expected_payload)
        if self._stale_read_pending and operation == self.case["target_op"]:
            self._stale_read_pending = False
            self._event("stale_snapshot_served", target_operation=operation)
            response = {"status": "not_found"}
        else:
            matches = [entry for entry in self.ledger
                       if entry["operation"] == operation and entry["payload"] == expected_payload]
            if matches:
                response = {"status": "found", "result": deepcopy(matches[0]["result"])}
            else:
                response = {"status": "not_found"}
        self._event("tool_return", operation="read_status", result=response)
        return response

    def settle(self) -> None:
        self._event("settle_started", pending_deliveries=len(self._schedule))
        while self._schedule:
            self._advance(self._schedule[0][0] - self.tick)
        self._event("settle_finished")


class _PolicyTools:
    """Expose tools without exposing case metadata or the oracle ledger."""

    def __init__(self, server: _ToolServer):
        self._mutate = server.mutate
        self._read = server.read_status
        self._wait = server.wait

    def mutate(self, operation: str, payload: dict[str, Any], operation_id: str | None = None) -> dict[str, Any]:
        return self._mutate(operation, payload, operation_id)

    def read_status(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._read(operation, payload)

    def wait(self) -> None:
        self._wait(1)


def _run_scripted_policy(tools: _PolicyTools, payload: dict[str, Any], policy: str,
                         request_namespace: str) -> bool:
    """Execute a bounded hand-written policy using only public observations."""
    def perform(operation: str, operation_payload: dict[str, Any]) -> dict[str, Any] | None:
        operation_id = f"{request_namespace}:{operation}" if policy == "idempotent_retry" else None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return tools.mutate(operation, operation_payload, operation_id)
            except ToolTimeout:
                if policy == "verify_then_retry":
                    status = tools.read_status(operation, operation_payload)
                    if status["status"] == "found":
                        return status["result"]
                if attempt + 1 < MAX_ATTEMPTS:
                    tools.wait()
        return None

    record_payload = {"task_id": payload["task_id"], "title": payload["title"], "body": payload["body"]}
    record = perform("create_record", record_payload)
    if record is None:
        return False
    notification_payload = {"task_id": payload["task_id"], "record_id": record["record_id"],
                            "recipient": payload["recipient"], "message": payload["message"]}
    notification = perform("send_notification", notification_payload)
    return notification is not None


def _score(ledger: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    records = [{**entry["payload"], **entry["result"]} for entry in ledger if entry["operation"] == "create_record"]
    notifications = [{**entry["payload"], **entry["result"]} for entry in ledger if entry["operation"] == "send_notification"]
    duplicates = max(0, len(records) - 1) + max(0, len(notifications) - 1)
    missing = int(not records) + int(not notifications)
    wrong_payload = 0
    valid_record_ids = set()
    for record in records:
        valid = (record.get("task_id") == payload["task_id"] and
                 record.get("title") == payload["title"] and record.get("body") == payload["body"])
        wrong_payload += int(not valid)
        if valid:
            valid_record_ids.add(record["record_id"])
    for notification in notifications:
        valid = (notification.get("task_id") == payload["task_id"] and
                 notification.get("recipient") == payload["recipient"] and
                 notification.get("message") == payload["message"] and
                 notification.get("record_id") in valid_record_ids)
        wrong_payload += int(not valid)
    return {"success": len(records) == 1 and len(notifications) == 1 and wrong_payload == 0,
            "duplicates": duplicates, "missing": missing, "wrong_payload": wrong_payload,
            "final_state": {"records": records, "notifications": notifications}}


def run_case(case: dict[str, Any], policy: str) -> dict[str, Any]:
    """Run one scripted baseline and score settled external effects."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    case = deepcopy(case)
    for key in ("task_id", "seed", "fault", "target_op", "payload"):
        if key not in case:
            raise ValueError(f"case missing {key}")
    if case["fault"] not in FAULTS:
        raise ValueError(f"fault must be one of {FAULTS}")
    if case["target_op"] not in ("create_record", "send_notification"):
        raise ValueError("target_op must be create_record or send_notification")
    if case["payload"].get("task_id") != case["task_id"]:
        raise ValueError("payload.task_id must match task_id")
    case.setdefault("case_id", f"{case['task_id']}-s{case['seed']}-{case['fault']}")
    case.setdefault("late_commit_delay", 4 + int(case["seed"]) % 3)
    if not isinstance(case["late_commit_delay"], int) or case["late_commit_delay"] < 1:
        raise ValueError("late_commit_delay must be a positive integer")
    server = _ToolServer(case)
    error = None
    try:
        # Fault, target, schedule, trace and ledger are not passed to the policy.
        claimed_success = _run_scripted_policy(_PolicyTools(server), deepcopy(case["payload"]),
                                               policy, case["task_id"])
    except (RuntimeError, ValueError, KeyError) as exc:
        claimed_success = False
        error = str(exc)
        server._event("policy_error", error=error)
    policy_ticks = server.tick
    server._event("policy_finished", claimed_success=claimed_success)
    server.settle()
    score = _score(server.ledger, case["payload"])
    result = {
        "case_id": case["case_id"], "task_id": case["task_id"], "seed": case["seed"],
        "fault": case["fault"], "target_op": case["target_op"], "policy": policy,
        "task_payload": deepcopy(case["payload"]),
        "condition_hash": _condition_hash(case),
        "evaluation_mode": "scripted_policy_simulation", "uses_language_model": False,
        "claimed_success": claimed_success, **score,
        "false_success": bool(claimed_success and not score["success"]),
        "tool_calls": server.tool_calls, "logical_ticks": server.tick,
        "policy_ticks": policy_ticks, "max_attempts_per_operation": MAX_ATTEMPTS,
        "trace": server.trace, "ledger": server.ledger,
    }
    if error is not None:
        result["error"] = error
    return result


def run_suite(cases: list[dict[str, Any]], policies: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Run the same case suite under each requested scripted policy."""
    selected = POLICIES if policies is None else policies
    if any(policy not in POLICIES for policy in selected):
        raise ValueError(f"policies must be selected from {POLICIES}")
    return [run_case(case, policy) for case in cases for policy in selected]


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "create_record",
        "description": "Create a record. Reusing operation_id with identical arguments deduplicates retries and delayed deliveries. A timeout does not disclose the write outcome.",
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"}, "title": {"type": "string"},
            "body": {"type": "string"}, "operation_id": {"type": "string"}},
            "required": ["task_id", "title", "body"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "send_notification",
        "description": "Send a notification linked to a record. Reusing operation_id with identical arguments deduplicates retries and delayed deliveries. A timeout does not disclose the write outcome.",
        "parameters": {"type": "object", "properties": {
            "task_id": {"type": "string"}, "record_id": {"type": "string"},
            "recipient": {"type": "string"}, "message": {"type": "string"},
            "operation_id": {"type": "string"}},
            "required": ["task_id", "record_id", "recipient", "message"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "read_status",
        "description": "Read the first matching result for an operation and exact business payload, excluding operation_id. A not_found response can be stale and does not cancel pending writes.",
        "parameters": {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["create_record", "send_notification"]},
            "payload": {"type": "object"}}, "required": ["operation", "payload"],
            "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "wait",
        "description": "Wait one logical simulation tick so delayed operations can progress.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]


class RecoveryEnvironment:
    """Adapter for external agents; never include privileged methods in tools.

    Only ``invoke`` and ``task`` belong on the agent-facing path. ``snapshot``,
    ``grade``, ``settle`` and ``finish`` are evaluator-only methods. A model
    runner can use TOOL_SCHEMAS as the OpenAI-compatible tool definitions.
    """

    def __init__(self, case: dict[str, Any]):
        self._case = deepcopy(case)
        self._case.setdefault("case_id", f"{case['task_id']}-s{case['seed']}-{case['fault']}")
        self._case.setdefault("late_commit_delay", 4 + int(case["seed"]) % 3)
        self._server = _ToolServer(self._case)
        self._finished = False

    @property
    def task(self) -> dict[str, Any]:
        return deepcopy(self._case["payload"])

    def invoke(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute one public tool without revealing injected fault metadata."""
        if self._finished:
            return {"ok": False, "error": {"type": "closed", "message": "episode has ended"}}
        if not isinstance(args, dict):
            return {"ok": False, "error": {"type": "invalid_arguments", "message": "arguments must be an object"}}
        try:
            if tool_name in ("create_record", "send_notification"):
                required = (("task_id", "title", "body") if tool_name == "create_record" else
                            ("task_id", "record_id", "recipient", "message"))
                unexpected = set(args) - set(required) - {"operation_id"}
                if unexpected or any(key not in args or not isinstance(args[key], str) for key in required):
                    raise ValueError("missing, invalid, or unexpected business arguments")
                operation_id = args.get("operation_id")
                if operation_id is not None and not isinstance(operation_id, str):
                    raise ValueError("operation_id must be a string")
                payload = {key: args[key] for key in required}
                result = self._server.mutate(tool_name, payload, operation_id)
            elif tool_name == "read_status":
                if (set(args) != {"operation", "payload"} or
                        args.get("operation") not in ("create_record", "send_notification") or
                        not isinstance(args.get("payload"), dict)):
                    raise ValueError("read_status requires operation and a payload object")
                result = self._server.read_status(args["operation"], args["payload"])
            elif tool_name == "wait":
                if args:
                    raise ValueError("wait takes no arguments")
                self._server.wait()
                result = {"waited_ticks": 1}
            else:
                raise ValueError("unknown tool")
            return {"ok": True, "result": result}
        except ToolTimeout as exc:
            return {"ok": False, "error": {"type": "timeout", "message": str(exc)}}
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            self._server._event("public_tool_error", operation=tool_name, error=str(exc))
            return {"ok": False, "error": {"type": "tool_error", "message": str(exc)}}

    def settle(self) -> None:
        """Evaluator only: deliver all pending effects before scoring."""
        self._server.settle()

    def snapshot(self) -> dict[str, Any]:
        """Evaluator only: inspect a detached copy of current business state."""
        return deepcopy(_score(self._server.ledger, self._case["payload"])["final_state"])

    def grade(self) -> dict[str, Any]:
        """Evaluator only: score the currently committed effects."""
        return deepcopy(_score(self._server.ledger, self._case["payload"]))

    def finish(self, claimed_success: bool, policy: str = "external_agent",
               evaluation_mode: str = "live_model") -> dict[str, Any]:
        """Evaluator only: close the episode, settle writes, and return a row."""
        if self._finished:
            raise RuntimeError("episode has already ended")
        self._finished = True
        policy_ticks = self._server.tick
        self._server._event("policy_finished", claimed_success=bool(claimed_success))
        self.settle()
        score = self.grade()
        return {"case_id": self._case["case_id"], "task_id": self._case["task_id"],
                "seed": self._case["seed"], "fault": self._case["fault"],
                "target_op": self._case["target_op"], "policy": policy,
                "task_payload": deepcopy(self._case["payload"]),
                "condition_hash": _condition_hash(self._case),
                "evaluation_mode": evaluation_mode, "uses_language_model": evaluation_mode == "live_model",
                "claimed_success": bool(claimed_success), **score,
                "false_success": bool(claimed_success and not score["success"]),
                "tool_calls": self._server.tool_calls, "logical_ticks": self._server.tick,
                "policy_ticks": policy_ticks, "trace": deepcopy(self._server.trace),
                "ledger": deepcopy(self._server.ledger)}
