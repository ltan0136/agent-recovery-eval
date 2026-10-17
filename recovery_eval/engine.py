"""Stateful, deterministic execution simulator and public-observation baselines.

No model inference occurs here. The four scripted policies are protocol
baselines; the public environment can also be driven by a real model runner.
The oracle settles late deliveries before independently checking business state.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import heapq
import json
from typing import Any

from .workflows import FAULTS, POLICIES, config_hash, workflow_specs


class _BusinessError(Exception):
    pass


class Environment:
    """Expose only public task, schemas, and tool returns to an agent.

    Oracle configuration and truth state are intentionally private attributes.
    Python attributes are not an adversarial security boundary; a malicious
    in-process policy must be isolated by its own process in production.
    """

    def __init__(self, case: dict[str, Any], *, max_calls: int = 24,
                 idempotency: bool | None = None):
        if type(max_calls) is not int or not 1 <= max_calls <= 1000:
            raise ValueError("max_calls must be an integer between 1 and 1000")
        self._case = deepcopy(case)
        self._spec = workflow_specs()[case["workflow"]]
        if case["fault"] not in FAULTS:
            raise ValueError("unknown fault")
        self._plan = [step.resolve(case["params"]) for step in self._spec.steps]
        self._by_tool = {step["tool"]: step for step in self._plan}
        self._by_name = {step["name"]: step for step in self._plan}
        if case["target_step"] not in self._by_name:
            raise ValueError("unknown target step")
        self._max_calls = max_calls
        self._idempotency = bool(case.get("idempotency_supported", True) if idempotency is None else idempotency)
        self._tick = 0
        self._calls = 0
        self._budget_exhausted = False
        self._fault_used = False
        self._stale_remaining = 0
        self._stale_snapshot: list[dict[str, Any]] | None = None
        self._ledger: list[dict[str, Any]] = []
        self._trace: list[dict[str, Any]] = []
        self._schedule: list[tuple[int, int, str, dict[str, Any], str | None]] = []
        self._request_results: dict[tuple[str, str], dict[str, Any]] = {}
        self._request_payloads: dict[tuple[str, str], dict[str, Any]] = {}
        self._scheduled_sequence = 0
        self._invalid_actions = 0
        self._finished: dict[str, Any] | None = None
        self._state = self._initial_state()
        self._config = {"case": {key: value for key, value in self._case.items() if key != "config_hash"},
                        "max_calls": max_calls, "idempotency_supported": self._idempotency,
                        "workflow_plan": deepcopy(self._plan),
                        "engine_version": "2.0", "settle_before_grade": True}

    @property
    def public_task(self) -> dict[str, Any]:
        task = self._spec.public_task(self._case["task_id"], self._case["params"], self._idempotency)
        task["tool_call_budget"] = self._max_calls
        return task

    @property
    def task(self) -> dict[str, Any]:
        return self.public_task

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for step in self._plan:
            properties = {}
            for name, value in step["arguments"].items():
                kind = "array" if isinstance(value, list) else "integer" if type(value) is int else "string"
                properties[name] = {"type": kind}
                if kind == "array":
                    properties[name]["items"] = {"type": "string"}
            properties["operation_id"] = {"type": "string", "description": "Stable key for this intended write; reuse on retries."}
            schemas.append({"type": "function", "function": {"name": step["tool"],
                "description": f"Execute step '{step['name']}'. Required prior steps: {', '.join(step['requires']) or 'none'}.",
                "parameters": {"type": "object", "properties": properties,
                               "required": list(step["arguments"]), "additionalProperties": False}}})
        schemas.append({"type": "function", "function": {"name": "inspect_step",
            "description": "Read committed evidence for a step. May be stale; absence does not cancel pending writes.",
            "parameters": {"type": "object", "properties": {"step": {"type": "string", "enum": list(self._by_name)}},
                           "required": ["step"], "additionalProperties": False}}})
        return schemas

    @property
    def trace(self) -> list[dict[str, Any]]:
        return deepcopy(self._trace)

    @property
    def ledger(self) -> list[dict[str, Any]]:
        return deepcopy(self._ledger)

    def _event(self, event: str, **fields: Any) -> None:
        self._trace.append({"index": len(self._trace), "tick": self._tick,
                            "event": event, **deepcopy(fields)})

    def _initial_state(self) -> dict[str, Any]:
        workflow = self._spec.name
        if workflow == "ticket_dispatch":
            return {"tickets": [], "assignments": [], "resolutions": [], "notifications": []}
        if workflow == "inventory_order":
            return {"initial_stock": self._case["params"]["initial_stock"],
                    "available_stock": self._case["params"]["initial_stock"],
                    "reservations": [], "orders": [], "charges": [], "shipments": [], "receipts": []}
        if workflow == "meeting_booking":
            return {"room_capacity": 1, "holds": [], "meetings": [], "invitations": [], "confirmations": []}
        return {"documents": [], "checks": [], "approvals": [], "publications": [], "notifications": []}

    def _advance(self, ticks: int = 1) -> None:
        target = self._tick + ticks
        while self._schedule and self._schedule[0][0] <= target:
            due, _, tool, arguments, operation_id = heapq.heappop(self._schedule)
            self._tick = due
            self._event("scheduled_delivery", tool=tool, operation_id=operation_id)
            self._commit(tool, arguments, operation_id, "late_delivery")
        self._tick = target

    def _error(self, kind: str, message: str, retryable: bool = False) -> dict[str, Any]:
        return {"ok": False, "error": {"type": kind, "message": message, "retryable": retryable}}

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._finished is not None:
            return self._error("finished", "The episode has already been scored.")
        if self._calls >= self._max_calls:
            self._budget_exhausted = True
            return self._error("budget_exhausted", "Tool-call budget exhausted.")
        self._advance()
        self._calls += 1
        self._event("tool_call", tool=name, arguments=arguments)
        if not isinstance(name, str):
            response = self._error("invalid_tool_name", "The tool name must be a string.")
        elif not isinstance(arguments, dict):
            response = self._error("invalid_arguments", "Arguments must be a JSON object.")
        elif name == "inspect_step":
            response = self._inspect(arguments)
        elif name not in self._by_tool:
            response = self._error("unknown_tool", "Unknown tool name.")
        else:
            response = self._mutate(name, deepcopy(arguments))
        self._event("tool_return", tool=name, response=response)
        return deepcopy(response)

    def _inspect(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if (set(arguments) != {"step"} or not isinstance(arguments.get("step"), str)
                or arguments["step"] not in self._by_name):
            return self._error("invalid_arguments", "Provide one registered step name.")
        visible = self._ledger
        if self._stale_remaining > 0 and arguments["step"] == self._case["target_step"]:
            self._stale_remaining -= 1
            visible = self._stale_snapshot or []
        effects = [effect for effect in visible if effect["step"] == arguments["step"]]
        return {"ok": True, "result": {"step": arguments["step"], "completed": bool(effects),
                                       "effect_count": len(effects),
                                       "evidence": [deepcopy(effect["result"]) for effect in effects]}}

    def _mutate(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        step = self._by_tool[tool]
        operation_id = arguments.pop("operation_id", None)
        if operation_id is not None and (not isinstance(operation_id, str) or not operation_id or len(operation_id) > 200):
            return self._error("invalid_arguments", "operation_id must be a nonempty string of at most 200 characters.")
        expected = step["arguments"]
        if set(arguments) != set(expected) or any(type(arguments[key]) is not type(value) for key, value in expected.items()):
            return self._error("invalid_arguments", "The tool arguments do not match its schema.")
        if any(isinstance(value, list) and any(not isinstance(item, str) for item in arguments[key])
               for key, value in expected.items()):
            return self._error("invalid_arguments", "Array arguments must contain only strings.")
        completed = {effect["step"] for effect in self._ledger}
        if any(prerequisite not in completed for prerequisite in step["requires"]):
            self._invalid_actions += 1
            return self._error("prerequisite_not_committed", "A required earlier operation has not committed.", True)
        key = (tool, operation_id) if self._idempotency and operation_id is not None else None
        if key is not None and key in self._request_payloads:
            if arguments != self._request_payloads[key]:
                return self._error("idempotency_conflict", "The operation key was already used with different arguments.")
            if key in self._request_results:
                self._event("request_deduplicated", tool=tool, operation_id=operation_id)
                return deepcopy(self._request_results[key])
            return self._error("timeout", "Request timed out; write outcome unknown.", True)
        target = step["name"] == self._case["target_step"]
        fault = self._case["fault"] if target and not self._fault_used else "clean"
        if target and self._case["fault"] == "permanent_failure":
            fault = "permanent_failure"
        if fault != "clean":
            self._fault_used = True
            self._event("fault_triggered", tool=tool, fault=fault)
        if fault == "pre_commit_timeout":
            return self._error("timeout", "Request timed out; write outcome unknown.", True)
        if fault == "transient_rate_limit":
            response = self._error("rate_limited", "Rate limit reached; retry after one logical call tick.", True)
            response["error"]["retry_after_ticks"] = 1
            return response
        if fault == "permanent_failure":
            return self._error("permanent_failure", "This operation is unavailable for the rest of this episode.")
        if fault == "late_commit":
            self._scheduled_sequence += 1
            heapq.heappush(self._schedule, (self._tick + self._case.get("late_commit_delay", 3),
                self._scheduled_sequence, tool, deepcopy(arguments), operation_id))
            if key is not None:
                self._request_payloads[key] = deepcopy(arguments)
            return self._error("timeout", "Request timed out; write outcome unknown.", True)
        if fault == "stale_read":
            self._stale_snapshot = deepcopy(self._ledger)
            self._stale_remaining = self._case.get("stale_reads", 2)
        response = self._commit(tool, arguments, operation_id, "tool_call")
        if response["ok"] and fault == "duplicate_delivery":
            self._commit(tool, arguments, operation_id, "gateway_redelivery")
        if response["ok"] and fault in {"lost_ack", "stale_read"}:
            return self._error("timeout", "Request timed out; write outcome unknown.", True)
        return response

    def _commit(self, tool: str, arguments: dict[str, Any], operation_id: str | None,
                source: str) -> dict[str, Any]:
        key = (tool, operation_id) if self._idempotency and operation_id is not None else None
        if key is not None and key in self._request_results:
            self._event("delivery_deduplicated", tool=tool, operation_id=operation_id, source=source)
            return deepcopy(self._request_results[key])
        step = self._by_tool[tool]
        completed = {effect["step"] for effect in self._ledger}
        if any(prerequisite not in completed for prerequisite in step["requires"]):
            self._invalid_actions += 1
            self._event("business_rejected", tool=tool, reason="prerequisite_not_committed", source=source)
            return self._error("prerequisite_not_committed", "A required earlier operation has not committed.", True)
        try:
            result = self._apply(tool, arguments)
        except _BusinessError as exc:
            self._invalid_actions += 1
            self._event("business_rejected", tool=tool, reason=str(exc), source=source)
            return self._error("business_constraint", str(exc))
        response = {"ok": True, "result": result}
        effect = {"effect_id": f"effect-{len(self._ledger) + 1:04d}", "tick": self._tick,
                  "step": step["name"], "tool": tool, "operation": tool,
                  "operation_id": operation_id, "arguments": deepcopy(arguments),
                  "result": deepcopy(result), "source": source}
        self._ledger.append(effect)
        self._event("effect_committed", **effect)
        if key is not None:
            self._request_payloads[key] = deepcopy(arguments)
            self._request_results[key] = deepcopy(response)
        return response

    def _append(self, collection: str, arguments: dict[str, Any], **extra: Any) -> dict[str, Any]:
        record = {"id": f"{collection}-{len(self._state[collection]) + 1:03d}",
                  **deepcopy(arguments), **deepcopy(extra)}
        self._state[collection].append(record)
        return {"id": record["id"], "collection": collection}

    def _require_ref(self, collection: str, key: str, value: Any) -> dict[str, Any]:
        for record in self._state[collection]:
            if record.get(key) == value:
                return record
        raise _BusinessError("Referenced resource does not exist.")

    def _apply(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        state = self._state
        if tool == "create_ticket":
            return self._append("tickets", args, status="open")
        if tool in {"assign_ticket", "resolve_ticket", "notify_ticket"}:
            ticket = self._require_ref("tickets", "ticket_ref", args["ticket_ref"])
            if tool == "assign_ticket":
                ticket["owner"] = args["owner"]
                ticket["status"] = "assigned"
                return self._append("assignments", args, ticket_id=ticket["id"])
            if tool == "resolve_ticket":
                if "owner" not in ticket:
                    raise _BusinessError("Ticket must have an owner before resolution.")
                ticket["resolution"] = args["resolution"]
                ticket["status"] = "resolved"
                return self._append("resolutions", args, ticket_id=ticket["id"])
            if ticket["status"] != "resolved":
                raise _BusinessError("Notification requires a resolved ticket.")
            return self._append("notifications", args, ticket_id=ticket["id"])
        if tool == "reserve_stock":
            if args["quantity"] < 1 or args["quantity"] > state["available_stock"]:
                raise _BusinessError("Insufficient stock or invalid quantity.")
            state["available_stock"] -= args["quantity"]
            return self._append("reservations", args)
        if tool == "create_order":
            if args["amount_cents"] <= 0:
                raise _BusinessError("Order amount must be positive.")
            return self._append("orders", args, status="created")
        if tool in {"charge_order", "ship_order", "send_receipt"}:
            order = self._require_ref("orders", "order_ref", args["order_ref"])
            if tool == "charge_order":
                if args["amount_cents"] != order["amount_cents"]:
                    raise _BusinessError("Charge amount differs from order amount.")
                order["status"] = "paid"
                return self._append("charges", args, order_id=order["id"])
            if tool == "ship_order":
                if order["status"] not in {"paid", "shipped"}:
                    raise _BusinessError("Order must be paid before shipment.")
                reservation = self._require_ref("reservations", "order_ref", args["order_ref"])
                order["status"] = "shipped"
                order["reservation_id"] = reservation["id"]
                return self._append("shipments", args, order_id=order["id"], reservation_id=reservation["id"])
            if order["status"] != "shipped" or args["amount_cents"] != order["amount_cents"]:
                raise _BusinessError("Receipt requires shipment and the correct order amount.")
            return self._append("receipts", args, order_id=order["id"])
        if tool == "hold_room":
            matching = [hold for hold in state["holds"] if hold["room"] == args["room"] and hold["slot"] == args["slot"]]
            if len(matching) >= state["room_capacity"]:
                raise _BusinessError("Room slot capacity exceeded.")
            return self._append("holds", args)
        if tool == "create_meeting":
            return self._append("meetings", args)
        if tool in {"invite_attendees", "confirm_booking"}:
            meeting = self._require_ref("meetings", "booking_ref", args["booking_ref"])
            if tool == "invite_attendees":
                if not args["attendees"] or len(args["attendees"]) != len(set(args["attendees"])):
                    raise _BusinessError("Attendees must be nonempty and unique.")
                return self._append("invitations", args, meeting_id=meeting["id"])
            invitation = self._require_ref("invitations", "booking_ref", args["booking_ref"])
            hold = self._require_ref("holds", "booking_ref", args["booking_ref"])
            return self._append("confirmations", args, meeting_id=meeting["id"], invitation_id=invitation["id"], hold_id=hold["id"])
        if tool == "submit_document":
            if not args["content"].strip() or args["revision"] < 1:
                raise _BusinessError("Document content and a positive revision are required.")
            return self._append("documents", args)
        if tool in {"check_document", "approve_document", "publish_document", "notify_subscribers"}:
            document = self._require_ref("documents", "document_ref", args["document_ref"])
            if document["revision"] != args["revision"]:
                raise _BusinessError("Operation revision does not match the submitted document.")
            if tool == "check_document":
                if set(args["checklist"]) != {"content_present", "revision_matches"}:
                    raise _BusinessError("The mandatory document checks are incomplete.")
                return self._append("checks", args, document_id=document["id"], passed=True)
            if tool == "approve_document":
                checks = self._require_ref("checks", "document_ref", args["document_ref"])
                if not checks["passed"] or not args["reviewer"]:
                    raise _BusinessError("Approval requires successful checks and a named reviewer.")
                return self._append("approvals", args, document_id=document["id"])
            if tool == "publish_document":
                approval = self._require_ref("approvals", "document_ref", args["document_ref"])
                if approval["revision"] != args["revision"]:
                    raise _BusinessError("Publication requires approval for this exact revision.")
                return self._append("publications", args, document_id=document["id"], approval_id=approval["id"])
            publication = self._require_ref("publications", "document_ref", args["document_ref"])
            return self._append("notifications", args, publication_id=publication["id"])
        raise _BusinessError("Unsupported business operation.")

    def _oracle(self) -> dict[str, Any]:
        counts = Counter(effect["step"] for effect in self._ledger)
        duplicates = sum(max(0, count - 1) for count in counts.values())
        missing_steps = [step["name"] for step in self._plan if counts[step["name"]] == 0]
        wrong_payload = sum(effect["arguments"] != self._by_name[effect["step"]]["arguments"] for effect in self._ledger)
        seen = set()
        order_violations = 0
        for effect in self._ledger:
            order_violations += sum(required not in seen for required in self._by_name[effect["step"]]["requires"])
            seen.add(effect["step"])
        params = self._case["params"]
        state = self._state
        resource_violations = 0
        business_violations = []
        collections = {
            "ticket_dispatch": {"create": "tickets", "assign": "assignments", "resolve": "resolutions", "notify": "notifications"},
            "inventory_order": {"reserve": "reservations", "order": "orders", "charge": "charges", "ship": "shipments", "receipt": "receipts"},
            "meeting_booking": {"hold": "holds", "meeting": "meetings", "invite": "invitations", "confirm": "confirmations"},
            "document_publication": {"submit": "documents", "check": "checks", "approve": "approvals", "publish": "publications", "notify": "notifications"},
        }[self._spec.name]
        for step_name, collection in collections.items():
            if len(state[collection]) != counts[step_name]:
                business_violations.append(f"{collection}_state_ledger_mismatch")
            expected_args = self._by_name[step_name]["arguments"]
            if any(any(record.get(key) != value for key, value in expected_args.items()) for record in state[collection]):
                business_violations.append(f"{collection}_content_mismatch")
        if self._spec.name == "ticket_dispatch":
            if counts["resolve"] and not all(ticket.get("status") == "resolved" and ticket.get("owner") == params["owner"]
                                              and ticket.get("resolution") == params["resolution"] for ticket in state["tickets"]):
                business_violations.append("ticket_final_state_mismatch")
            if state["tickets"] and any(item["ticket_id"] not in {ticket["id"] for ticket in state["tickets"]}
                                        for collection in ("assignments", "resolutions", "notifications") for item in state[collection]):
                business_violations.append("broken_ticket_reference")
        elif self._spec.name == "inventory_order":
            reserved = sum(item["quantity"] for item in state["reservations"])
            charged = sum(item["amount_cents"] for item in state["charges"])
            resource_violations += int(state["available_stock"] < 0 or reserved > params["quantity"])
            resource_violations += int(charged > params["amount_cents"])
            if state["available_stock"] != state["initial_stock"] - reserved:
                business_violations.append("inventory_conservation_failure")
            if counts["ship"] and any(order["status"] != "shipped" for order in state["orders"]):
                business_violations.append("order_final_state_mismatch")
            reservation_ids = {item["id"] for item in state["reservations"]}
            order_ids = {item["id"] for item in state["orders"]}
            if any(item["reservation_id"] not in reservation_ids or item["order_id"] not in order_ids for item in state["shipments"]):
                business_violations.append("shipment_resource_reference_mismatch")
        elif self._spec.name == "meeting_booking":
            occupied = Counter((item["room"], item["slot"]) for item in state["holds"])
            resource_violations += sum(max(0, count - state["room_capacity"]) for count in occupied.values())
            if counts["confirm"] and len(state["invitations"]) != 1:
                business_violations.append("confirmation_requires_one_invitation_batch")
            for confirmation in state["confirmations"]:
                for key, collection in (("hold_id", "holds"), ("meeting_id", "meetings"), ("invitation_id", "invitations")):
                    if confirmation[key] not in {item["id"] for item in state[collection]}:
                        business_violations.append("confirmation_resource_reference_mismatch")
        else:
            approved = {(item["document_ref"], item["revision"], item["id"]) for item in state["approvals"]}
            for publication in state["publications"]:
                if (publication["document_ref"], publication["revision"], publication["approval_id"]) not in approved:
                    business_violations.append("publication_without_matching_approval")
            if counts["check"] and not all(item["passed"] for item in state["checks"]):
                business_violations.append("failed_document_checks")
        reasons = []
        for count, name in ((duplicates, "duplicate_effects"), (len(missing_steps), "missing_steps"),
                            (wrong_payload, "wrong_payload"), (order_violations, "order_violations"),
                            (resource_violations, "resource_violations")):
            if count:
                reasons.append(name)
        reasons.extend(business_violations)
        return {"success": not reasons, "passed": not reasons,
                "task_complete": not missing_steps, "duplicates": duplicates, "duplicate_effects": duplicates,
                "missing": len(missing_steps), "missing_steps": missing_steps,
                "wrong_payload": wrong_payload, "order_violations": order_violations,
                "resource_violations": resource_violations, "business_violations": business_violations,
                "failure_reasons": reasons}

    def finish(self, claimed_success: bool, agent_id: str = "unknown", agent_kind: str = "scripted",
               **metadata: Any) -> dict[str, Any]:
        if self._finished is not None:
            return deepcopy(self._finished)
        if type(claimed_success) is not bool:
            raise ValueError("claimed_success must be a boolean")
        if self._schedule:
            self._advance(max(item[0] for item in self._schedule) - self._tick)
        before = json.dumps({"state": self._state, "ledger": self._ledger}, sort_keys=True)
        score = self._oracle()
        after = json.dumps({"state": self._state, "ledger": self._ledger}, sort_keys=True)
        if before != after:
            raise RuntimeError("Oracle must not mutate the business state or ledger")
        full_hash = config_hash(self._config)
        row = {"schema_version": "2.0", "case_id": self._case["case_id"],
               "task_id": self._case["task_id"], "workflow": self._spec.name,
               "seed": self._case["seed"], "variant": self._case.get("variant", 0),
               "variant_id": self._case.get("variant_id", self._case["task_id"]),
               "fault": self._case["fault"], "target_step": self._case["target_step"],
               "agent_id": agent_id, "agent_kind": agent_kind, "policy": agent_id,
               "claimed_success": claimed_success, "false_success": claimed_success and not score["success"],
               "false_failure": not claimed_success and score["success"],
               "tool_calls": self._calls, "logical_ticks": self._tick,
               "invalid_actions": self._invalid_actions,
               "budget_exhausted": self._budget_exhausted,
               "public_task": self.public_task, "trace": deepcopy(self._trace),
               "ledger": deepcopy(self._ledger), "final_state": deepcopy(self._state),
               "case_config": deepcopy(self._case), "execution_config": deepcopy(self._config),
               "case_hash": config_hash(self._case), "config_hash": full_hash,
               "condition_hash": full_hash, "oracle_read_only": True, "settled_before_grade": True,
               **score}
        if set(metadata) & set(row):
            raise ValueError("Metadata cannot overwrite scored result fields")
        row.update(metadata)
        self._finished = deepcopy(row)
        return row


def run_scripted(case: dict[str, Any], policy: str = "idempotent_retry", *,
                 max_calls: int = 24, idempotency: bool | None = None) -> dict[str, Any]:
    """Run a generic baseline using only public plan and observed tool returns."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    environment = Environment(case, max_calls=max_calls, idempotency=idempotency)
    task = environment.public_task
    claimed = True
    for step in task["steps"]:
        arguments = deepcopy(step["arguments"])
        if policy == "idempotent_retry":
            arguments["operation_id"] = f"{task['task_id']}:{step['name']}"
        completed = False
        attempts = 1 if policy == "no_retry" else 6
        for _ in range(attempts):
            response = environment.invoke(step["tool"], arguments)
            if response["ok"]:
                completed = True
                break
            if not response.get("error", {}).get("retryable", False):
                break
            if policy == "verify_then_retry":
                observed = environment.invoke("inspect_step", {"step": step["name"]})
                if observed["ok"] and observed["result"]["completed"]:
                    completed = True
                    break
        if not completed:
            claimed = False
            break
    return environment.finish(claimed, policy, "scripted")
