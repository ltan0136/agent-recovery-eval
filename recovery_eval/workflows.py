"""Declarative, stateful business workflows for recovery experiments.

The public step plan deliberately removes planning difficulty. This benchmark
evaluates recovery and execution, not discovery of a hidden workflow.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable


FAULTS = (
    "clean", "pre_commit_timeout", "lost_ack", "late_commit", "stale_read",
    "duplicate_delivery", "transient_rate_limit", "permanent_failure",
)
POLICIES = ("no_retry", "bounded_retry", "verify_then_retry", "idempotent_retry")


def config_hash(value: dict[str, Any]) -> str:
    """Hash all configuration fields except an existing self-referential hash."""
    clean = {key: value[key] for key in value if key not in {"config_hash", "condition_hash"}}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Step:
    name: str
    tool: str
    requires: tuple[str, ...]
    arguments: dict[str, Any]

    def resolve(self, params: dict[str, Any]) -> dict[str, Any]:
        arguments = {}
        for key, value in self.arguments.items():
            if isinstance(value, str) and value.startswith("$"):
                arguments[key] = deepcopy(params[value[1:]])
            else:
                arguments[key] = deepcopy(value)
        return {"name": self.name, "tool": self.tool,
                "requires": list(self.requires), "arguments": arguments}


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    title: str
    description: str
    steps: tuple[Step, ...]

    def public_task(self, task_id: str, params: dict[str, Any],
                    idempotency_supported: bool = True) -> dict[str, Any]:
        return {
            "task_id": task_id, "workflow_id": self.name,
            "title": self.title, "goal": self.description,
            "params": deepcopy(params),
            "steps": [step.resolve(params) for step in self.steps],
            "plan_provided": True,
            "service_contract": {
                "idempotency_supported": idempotency_supported,
                "operation_id": "Optional stable key per intended write; reuse unchanged on retries.",
                "timeouts": "Unknown write outcome. A timeout does not cancel a pending write.",
                "reads": "inspect_step may be stale; not_found is not proof of cancellation.",
                "prerequisites": "The server rejects writes whose prerequisites have not committed.",
            },
            "verification": {"tool": "inspect_step", "arguments": {"step": "<step name>"}},
        }


def workflow_specs() -> dict[str, WorkflowSpec]:
    """Return new specifications so callers cannot mutate a global registry."""
    return {
        "ticket_dispatch": WorkflowSpec(
            "ticket_dispatch", "Resolve a support ticket and notify its owner",
            "Create exactly one correct ticket, assign and resolve it, then notify the requested recipient once.",
            (
                Step("create", "create_ticket", (), {"ticket_ref": "$ticket_ref", "title": "$title", "severity": "$severity"}),
                Step("assign", "assign_ticket", ("create",), {"ticket_ref": "$ticket_ref", "owner": "$owner"}),
                Step("resolve", "resolve_ticket", ("assign",), {"ticket_ref": "$ticket_ref", "resolution": "$resolution"}),
                Step("notify", "notify_ticket", ("resolve",), {"ticket_ref": "$ticket_ref", "recipient": "$recipient", "message": "$message"}),
            ),
        ),
        "inventory_order": WorkflowSpec(
            "inventory_order", "Reserve, charge, and ship an inventory order",
            "Reserve the requested stock once, create and charge one order, ship it, and issue one receipt without excess resource use.",
            (
                Step("reserve", "reserve_stock", (), {"order_ref": "$order_ref", "sku": "$sku", "quantity": "$quantity"}),
                Step("order", "create_order", (), {"order_ref": "$order_ref", "customer": "$customer", "amount_cents": "$amount_cents"}),
                Step("charge", "charge_order", ("order",), {"order_ref": "$order_ref", "amount_cents": "$amount_cents"}),
                Step("ship", "ship_order", ("reserve", "charge"), {"order_ref": "$order_ref", "address": "$address"}),
                Step("receipt", "send_receipt", ("ship",), {"order_ref": "$order_ref", "recipient": "$recipient", "amount_cents": "$amount_cents"}),
            ),
        ),
        "meeting_booking": WorkflowSpec(
            "meeting_booking", "Book a room and invite meeting attendees",
            "Hold one room slot, create one meeting, invite the named attendees once, and confirm only after invitations.",
            (
                Step("hold", "hold_room", (), {"booking_ref": "$booking_ref", "room": "$room", "slot": "$slot"}),
                Step("meeting", "create_meeting", (), {"booking_ref": "$booking_ref", "title": "$title", "agenda": "$agenda"}),
                Step("invite", "invite_attendees", ("meeting",), {"booking_ref": "$booking_ref", "attendees": "$attendees"}),
                Step("confirm", "confirm_booking", ("hold", "invite"), {"booking_ref": "$booking_ref", "organizer": "$organizer"}),
            ),
        ),
        "document_publication": WorkflowSpec(
            "document_publication", "Review, approve, and publish a document",
            "Submit the exact document, complete checks and reviewer approval, publish the approved revision once, then notify subscribers once.",
            (
                Step("submit", "submit_document", (), {"document_ref": "$document_ref", "revision": "$revision", "content": "$content"}),
                Step("check", "check_document", ("submit",), {"document_ref": "$document_ref", "revision": "$revision", "checklist": "$checklist"}),
                Step("approve", "approve_document", ("submit", "check"), {"document_ref": "$document_ref", "revision": "$revision", "reviewer": "$reviewer"}),
                Step("publish", "publish_document", ("approve",), {"document_ref": "$document_ref", "revision": "$revision", "channel": "$channel"}),
                Step("notify", "notify_subscribers", ("publish",), {"document_ref": "$document_ref", "revision": "$revision", "audience": "$audience"}),
            ),
        ),
    }


def _parameters(workflow: str, variant: int) -> dict[str, Any]:
    number = variant + 1
    if workflow == "ticket_dispatch":
        return {"ticket_ref": f"T-{number:03d}", "title": f"Restore service {number}",
                "severity": ("low", "medium", "high", "critical")[variant % 4],
                "owner": f"engineer-{number}", "resolution": f"Restarted verified service {number}",
                "recipient": f"requester-{number}@example.test", "message": f"Ticket T-{number:03d} resolved."}
    if workflow == "inventory_order":
        quantity = variant % 3 + 1
        return {"order_ref": f"O-{number:03d}", "sku": f"SKU-{number:03d}", "quantity": quantity,
                "initial_stock": quantity * 3, "customer": f"customer-{number}",
                "amount_cents": quantity * (1000 + number * 75), "address": f"{number} Example Street",
                "recipient": f"buyer-{number}@example.test"}
    if workflow == "meeting_booking":
        return {"booking_ref": f"M-{number:03d}", "room": f"Room-{number}",
                "slot": f"2027-01-{variant % 27 + 1:02d}T10:00:00Z", "title": f"Review {number}",
                "agenda": f"Review release {number} and assign follow-up work.",
                "attendees": [f"member-{number}@example.test", f"reviewer-{number}@example.test"],
                "organizer": f"organizer-{number}@example.test"}
    return {"document_ref": f"D-{number:03d}", "revision": number,
            "content": f"Release procedure revision {number}: review, approve, then publish.",
            "checklist": ["content_present", "revision_matches"], "reviewer": f"reviewer-{number}",
            "channel": ("internal", "public")[variant % 2], "audience": f"subscribers-{number}"}


def build_matrix(workflows: Iterable[str] | str | None = None, variants: int = 4,
                 seeds: int = 3, faults: Iterable[str] | str | None = None) -> list[dict[str, Any]]:
    """Build a factorial suite; task/seed is a paired cluster across conditions.

    Variants change public task parameters. Seeds change target steps, stale
    visibility, and delayed delivery schedules. Repeated deterministic cases
    must not be presented as independent language-model samples.
    """
    for name, value in (("variants", variants), ("seeds", seeds)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    registry = workflow_specs()
    selected = list(registry) if workflows is None else ([workflows] if isinstance(workflows, str) else list(workflows))
    selected_faults = list(FAULTS) if faults is None else ([faults] if isinstance(faults, str) else list(faults))
    if not selected or any(name not in registry for name in selected) or len(selected) != len(set(selected)):
        raise ValueError("workflows must contain distinct registered workflow names")
    if not selected_faults or any(fault not in FAULTS for fault in selected_faults) or len(selected_faults) != len(set(selected_faults)):
        raise ValueError("faults must contain distinct registered condition names")
    cases = []
    for workflow in selected:
        spec = registry[workflow]
        for variant in range(variants):
            task_id = f"{workflow}-v{variant + 1:03d}"
            for seed in range(seeds):
                target = spec.steps[(variant + seed) % len(spec.steps)].name
                for fault in selected_faults:
                    case = {"schema_version": "2.0", "case_id": f"{task_id}-s{seed}-{fault}",
                            "task_id": task_id, "variant_id": task_id,
                            "workflow": workflow, "variant": variant,
                            "params": _parameters(workflow, variant), "seed": seed,
                            "fault": fault, "target_step": target,
                            "late_commit_delay": 3 + seed % 3,
                            "stale_reads": 2 + seed % 2,
                            "idempotency_supported": True}
                    case["config_hash"] = config_hash(case)
                    cases.append(case)
    return cases
