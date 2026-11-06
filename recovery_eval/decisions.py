"""Decision-level and paired recovery metrics derived from saved episodes.

Both metrics read only fields every v0.2 episode row already stores (the public
trace, the public task and the scored outcome), so they can be recomputed for
runs written before they existed.

Retry permission
    Final-state scoring cannot tell a safe recovery from a lucky one. Following
    the separation in RetryBench (arXiv:2610.15884), every write that repeats a
    step whose earlier attempt returned ``timeout`` (outcome unknown) is a
    *retry decision*, and it is *justified* only if the public contract
    guarantees it cannot duplicate the effect. Under this simulator's public
    service contract a timeout does not cancel a pending write and
    ``inspect_step`` may be stale, so a ``not_found`` read never proves
    non-execution. The only guarantee is therefore an idempotency key: the
    service must advertise ``idempotency_supported`` and the retry must reuse
    the same non-empty ``operation_id`` as every earlier unresolved attempt of
    that step. Writes repeated after an error that proves non-execution
    (rate limit, rejected prerequisite, invalid arguments, ...) are not retry
    decisions, because there is nothing uncertain to justify.

Conditional recovery success rate (CRSR)
    UndoBench (arXiv:2610.05622) separates nominal competence from recovery by
    pairing each faulted episode with the clean episode of the same task and
    seed. Here a faulted episode is eligible when the same agent succeeded on
    the clean condition of the same (workflow, variant, seed); CRSR is the
    success rate among eligible episodes. ``permanent_failure`` makes the
    target step unreachable by design, so it is reported on its own but left
    out of the ``recoverable_faults`` aggregate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

UNKNOWN_OUTCOME_ERRORS = frozenset({"timeout"})
UNRECOVERABLE_FAULTS = frozenset({"permanent_failure"})


def _idempotency_supported(row: dict[str, Any]) -> bool:
    contract = (row.get("public_task") or {}).get("service_contract") or {}
    return contract.get("idempotency_supported") is True


def retry_audit(row: dict[str, Any]) -> dict[str, int] | None:
    """Count retry decisions after unknown write outcomes in one episode's public trace.

    Returns None when the row has no public trace or task (for example rows
    produced by unit-test doubles), so callers can skip it.
    """
    trace, task = row.get("trace"), row.get("public_task")
    if not isinstance(trace, list) or not isinstance(task, dict):
        return None
    write_tools = {step["tool"] for step in task.get("steps", [])}
    idempotent = _idempotency_supported(row)
    unresolved: dict[str, list[str | None]] = {}  # tool -> keys of timed-out attempts
    confirmed: set[str] = set()
    counts = {"uncertain_writes": 0, "retry_decisions": 0,
              "justified_retries": 0, "unjustified_retries": 0}
    pending: dict[str, Any] | None = None
    for event in trace:
        if event.get("event") == "tool_call":
            pending = event
            continue
        if event.get("event") != "tool_return" or pending is None:
            continue
        call, pending = pending, None
        tool, arguments, response = call.get("tool"), call.get("arguments"), event.get("response") or {}
        if tool not in write_tools or not isinstance(arguments, dict):
            continue
        key = arguments.get("operation_id")
        key = key if isinstance(key, str) and key else None
        if tool in unresolved and tool not in confirmed:
            counts["retry_decisions"] += 1
            if idempotent and key is not None and all(prior == key for prior in unresolved[tool]):
                counts["justified_retries"] += 1
            else:
                counts["unjustified_retries"] += 1
        if response.get("ok") is True:
            confirmed.add(tool)
        elif (response.get("error") or {}).get("type") in UNKNOWN_OUTCOME_ERRORS:
            counts["uncertain_writes"] += 1
            unresolved.setdefault(tool, []).append(key)
    return counts


def retry_permission(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate retry decisions per agent, overall and per condition."""
    slices: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, int]]]] = defaultdict(list)
    for row in rows:
        audit = retry_audit(row)
        if audit is None:
            continue
        slices[(row["agent_id"], "all")].append((row, audit))
        slices[(row["agent_id"], row["fault"])].append((row, audit))
    result = []
    for (agent, fault), items in sorted(slices.items()):
        episodes = len(items)
        decisions = sum(a["retry_decisions"] for _, a in items)
        justified = sum(a["justified_retries"] for _, a in items)
        successes = sum(bool(r["success"]) for r, _ in items)
        lucky = sum(bool(r["success"]) and a["unjustified_retries"] > 0 for r, a in items)
        result.append({
            "agent_id": agent, "fault": fault, "episodes": episodes,
            "episodes_with_uncertain_write": sum(a["uncertain_writes"] > 0 for _, a in items),
            "retry_decisions": decisions, "justified_retries": justified,
            "unjustified_retries": decisions - justified,
            "justified_retry_share": justified / decisions if decisions else None,
            "episodes_with_unjustified_retry": sum(a["unjustified_retries"] > 0 for _, a in items),
            "successes": successes,
            "successes_with_unjustified_retry": lucky,
            "success_without_unjustified_retry_rate": (successes - lucky) / episodes,
        })
    return result


def conditional_recovery(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Success on faulted episodes whose paired clean episode succeeded (CRSR)."""
    clean_success: set[tuple[str, str, str, int]] = set()
    for row in rows:
        if row["fault"] == "clean" and row["success"]:
            clean_success.add((row["agent_id"], row["workflow"], row["variant_id"], row["seed"]))
    slices: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["fault"] == "clean":
            continue
        slices[(row["agent_id"], row["fault"])].append(row)
        if row["fault"] not in UNRECOVERABLE_FAULTS:
            slices[(row["agent_id"], "recoverable_faults")].append(row)
    result = []
    for (agent, fault), items in sorted(slices.items()):
        eligible = [r for r in items
                    if (agent, r["workflow"], r["variant_id"], r["seed"]) in clean_success]
        successes = sum(bool(r["success"]) for r in eligible)
        result.append({"agent_id": agent, "fault": fault, "faulted_episodes": len(items),
                       "eligible_episodes": len(eligible), "successes": successes,
                       "conditional_recovery_success_rate": successes / len(eligible) if eligible else None})
    return result
