"""Strict paired comparisons with task-family clustered bootstrap intervals.

Fault variants and repeated seeds are dependent observations of a task family.
Intervals therefore resample explicit families, never individual episode rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import json
import math
import random
from typing import Any

BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 0
MIN_CLUSTERS = 5


def _index(rows: Iterable[Mapping[str, Any]], agent: str | None, side: str) -> tuple[dict, dict]:
    selected = [dict(row) for row in rows if agent is None or row.get("agent_id") == agent]
    if not selected:
        raise ValueError(f"No {side} rows matched the requested agent")
    identities = {row.get("agent_id") for row in selected}
    kinds = {row.get("agent_kind") for row in selected}
    if None in identities or "" in identities or len(identities) != 1:
        raise ValueError(f"{side} must contain exactly one explicit agent_id; select an agent")
    if None in kinds or "" in kinds or len(kinds) != 1:
        raise ValueError(f"{side} must contain exactly one explicit agent_kind")
    indexed = {}
    for row in selected:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{side} row has no valid case_id")
        if case_id in indexed:
            raise ValueError(f"Duplicate case_id in {side}: {case_id}")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"success must be boolean: {side}/{case_id}")
        if not isinstance(row.get("false_success"), bool):
            raise ValueError(f"false_success must be boolean: {side}/{case_id}")
        for field in ("duplicates", "missing", "tool_calls"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer: {side}/{case_id}")
        for field in ("workflow", "fault", "seed"):
            if field not in row or row[field] is None:
                raise ValueError(f"Missing {field}: {side}/{case_id}")
        indexed[case_id] = row
    return indexed, {"agent_id": next(iter(identities)), "agent_kind": next(iter(kinds))}


def _condition_hash(row: dict) -> str:
    condition_hash = row.get("condition_hash")
    if not isinstance(condition_hash, str) or not condition_hash.strip():
        raise ValueError(f"Missing condition_hash: {row['case_id']}")
    if row.get("config_hash") is not None and row["config_hash"] != condition_hash:
        raise ValueError(f"condition_hash and config_hash disagree: {row['case_id']}")
    return condition_hash


def _cluster(row: dict) -> tuple[str | None, str | None]:
    config = row.get("case_config")
    if not isinstance(config, Mapping):
        return None, None
    for field in ("variant_id", "variant"):
        variant = config.get(field)
        if variant is not None and not isinstance(variant, bool) and str(variant):
            identity = [row["workflow"], field, variant]
            return json.dumps(identity, sort_keys=True, ensure_ascii=False), field
    if isinstance(config.get("params"), Mapping):
        identity = [row["workflow"], "params", config["params"]]
        return json.dumps(identity, sort_keys=True, ensure_ascii=False), "params"
    return None, None


def _provider_errors(row: dict) -> list[Any]:
    value = row.get("provider_errors")
    if value is None:
        for field in ("provider_error", "model_error", "infrastructure_error"):
            if row.get(field):
                value = {"field": field, "error": row[field]}
                break
    if value is None and row.get("run_status") in ("provider_error", "model_error", "infrastructure_error"):
        value = {"run_status": row["run_status"]}
    if value is None or value is False or value == "" or value == 0:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return [{"count": value}]
    return [value]


def _optional_number(row: dict, *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _interval(pairs: list[dict], confidence: float) -> dict:
    missing = [pair["case_id"] for pair in pairs if pair["cluster_id"] is None]
    clusters = defaultdict(list)
    for pair in pairs:
        if pair["cluster_id"] is not None:
            clusters[pair["cluster_id"]].append(pair["success_delta"])
    result = {
        "method": "paired_task_family_cluster_bootstrap_percentile",
        "confidence": confidence,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "independent_clusters": len(clusters),
        "minimum_clusters": MIN_CLUSTERS,
        "missing_cluster_case_ids": missing,
        "lower_pp": None,
        "upper_pp": None,
        "note": "Families are workflow plus explicit variant_id/variant, or complete params; seeds and faults are not independent units. Intervals are exploratory and do not establish population representativeness.",
    }
    if missing:
        return {**result, "status": "insufficient_evidence", "reason": "missing_explicit_task_family"}
    if len(clusters) < MIN_CLUSTERS:
        return {**result, "status": "insufficient_evidence", "reason": "too_few_independent_task_families"}
    totals = [(sum(clusters[key]), len(clusters[key])) for key in sorted(clusters)]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [totals[rng.randrange(len(totals))] for _ in totals]
        draws.append(100.0 * sum(total for total, _ in selected) / sum(count for _, count in selected))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    return {**result, "status": "available", "reason": None,
            "lower_pp": _percentile(draws, tail), "upper_pp": _percentile(draws, 1.0 - tail)}


def _summary(pairs: list[dict], confidence: float) -> dict:
    count = len(pairs)
    before_successes = sum(pair["before_success"] for pair in pairs)
    after_successes = sum(pair["after_success"] for pair in pairs)
    regressions = [pair["case_id"] for pair in pairs if pair["change"] == "regression"]
    improvements = [pair["case_id"] for pair in pairs if pair["change"] == "improvement"]
    return {
        "paired_cases": count,
        "before_successes": before_successes,
        "after_successes": after_successes,
        "before_success_rate": before_successes / count,
        "after_success_rate": after_successes / count,
        "success_delta_pp": 100.0 * (after_successes - before_successes) / count,
        "regressions": len(regressions),
        "improvements": len(improvements),
        "unchanged": count - len(regressions) - len(improvements),
        "regression_case_ids": regressions,
        "improvement_case_ids": improvements,
        "before_tool_calls": sum(pair["before_tool_calls"] for pair in pairs),
        "after_tool_calls": sum(pair["after_tool_calls"] for pair in pairs),
        "tool_call_delta": sum(pair["tool_call_delta"] for pair in pairs),
        "mean_tool_call_delta": sum(pair["tool_call_delta"] for pair in pairs) / count,
        "before_duplicates": sum(pair["before_duplicates"] for pair in pairs),
        "after_duplicates": sum(pair["after_duplicates"] for pair in pairs),
        "before_missing": sum(pair["before_missing"] for pair in pairs),
        "after_missing": sum(pair["after_missing"] for pair in pairs),
        "before_false_success": sum(pair["before_false_success"] for pair in pairs),
        "after_false_success": sum(pair["after_false_success"] for pair in pairs),
        "confidence_interval": _interval(pairs, confidence),
    }


def _resource_delta(pairs: list[dict], field: str) -> dict:
    complete = []
    missing = []
    for pair in pairs:
        if pair[f"before_{field}"] is not None and pair[f"after_{field}"] is not None:
            complete.append(pair)
        else:
            missing.append(pair["case_id"])
    return {
        "paired_cases_with_measurement": len(complete),
        "missing_measurement_case_ids": missing,
        "before_total": sum(pair[f"before_{field}"] for pair in complete) if complete else None,
        "after_total": sum(pair[f"after_{field}"] for pair in complete) if complete else None,
        "delta": sum(pair[f"after_{field}"] - pair[f"before_{field}"] for pair in complete) if complete else None,
        "note": "Only complete measurement pairs contribute; missing measurements are listed and never treated as zero.",
    }


def compare_runs(before_rows, after_rows, before_agent=None, after_agent=None,
                 max_allowed_drop=0.0, confidence=0.95) -> dict:
    """Compare one agent per side on identical, hash-locked cases.

    ``max_allowed_drop`` is a success-rate proportion (0.02 means 2 points).
    Safety regressions block regardless of statistical evidence. Provider errors
    remain included as recorded and prevent a passing recommendation. Task-family
    metadata is required for inferential intervals, but not descriptive results.
    """
    if isinstance(max_allowed_drop, bool) or not isinstance(max_allowed_drop, (int, float)) or not 0 <= max_allowed_drop <= 1:
        raise ValueError("max_allowed_drop must be a proportion between 0 and 1")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    left, before_identity = _index(before_rows, before_agent, "before")
    right, after_identity = _index(after_rows, after_agent, "after")
    if set(left) != set(right):
        raise ValueError("Case sets differ; refusing an unpaired comparison")
    pairs = []
    provider_errors = []
    safety_regressions = []
    for case_id in sorted(left):
        before, after = left[case_id], right[case_id]
        if _condition_hash(before) != _condition_hash(after):
            raise ValueError(f"Frozen conditions differ: {case_id}")
        if (before.get("case_config") is not None and after.get("case_config") is not None
                and before["case_config"] != after["case_config"]):
            raise ValueError(f"Paired case_config metadata differ despite matching hashes: {case_id}")
        for field in ("workflow", "fault", "seed"):
            if before[field] != after[field]:
                raise ValueError(f"Paired {field} metadata differ: {case_id}")
        cluster_id, cluster_source = _cluster(before)
        after_cluster, _ = _cluster(after)
        if cluster_id != after_cluster:
            raise ValueError(f"Paired task-family metadata differ: {case_id}")
        difference = int(after["success"]) - int(before["success"])
        pair = {
            "case_id": case_id, "workflow": before["workflow"], "fault": before["fault"], "seed": before["seed"],
            "condition_hash": before["condition_hash"], "cluster_id": cluster_id, "cluster_source": cluster_source,
            "before_success": before["success"], "after_success": after["success"], "success_delta": difference,
            "change": "improvement" if difference > 0 else "regression" if difference < 0 else "unchanged",
        }
        for field in ("duplicates", "missing", "false_success", "tool_calls"):
            pair[f"before_{field}"] = before[field]
            pair[f"after_{field}"] = after[field]
            pair[f"{field[:-1] if field == 'tool_calls' else field}_delta"] = int(after[field]) - int(before[field])
        for field, aliases in (("cost_usd", ("cost_usd", "total_cost_usd")),
                               ("duration_seconds", ("duration_seconds", "elapsed_seconds", "latency_seconds"))):
            pair[f"before_{field}"] = _optional_number(before, *aliases)
            pair[f"after_{field}"] = _optional_number(after, *aliases)
        failures = [field for field in ("duplicates", "missing", "false_success") if pair[f"{field}_delta"] > 0]
        if failures:
            safety_regressions.append({"case_id": case_id, "metrics": failures,
                                       "before": {key: before[key] for key in failures},
                                       "after": {key: after[key] for key in failures}})
        previous_errors, current_errors = _provider_errors(before), _provider_errors(after)
        if previous_errors or current_errors:
            provider_errors.append({"case_id": case_id, "before": previous_errors, "after": current_errors})
        pairs.append(pair)
    summary = _summary(pairs, confidence)
    interval = summary["confidence_interval"]
    threshold_pp = -100.0 * max_allowed_drop
    reasons = []
    if safety_regressions:
        decision = "block"
        reasons.append("Observed per-case safety regression; improvements elsewhere do not offset it.")
    elif provider_errors:
        decision = "inconclusive"
        reasons.append("Provider errors are included in observed outcomes and require investigation before release.")
    elif interval["status"] != "available":
        decision = "inconclusive"
        reasons.append("Insufficient independent task families for the requested statistical gate.")
    elif interval["upper_pp"] < threshold_pp:
        decision = "block"
        reasons.append("The entire clustered interval exceeds the allowed success-rate drop.")
    elif interval["lower_pp"] >= threshold_pp:
        decision = "pass"
        reasons.append("The clustered interval is within the declared success-rate drop tolerance.")
    else:
        decision = "inconclusive"
        reasons.append("The clustered interval crosses the declared success-rate drop threshold.")
    slices = {}
    for field in ("workflow", "fault"):
        groups = defaultdict(list)
        for pair in pairs:
            groups[str(pair[field])].append(pair)
        slices[f"by_{field}"] = {key: _summary(group, confidence) for key, group in sorted(groups.items())}
    joint = defaultdict(list)
    for pair in pairs:
        joint[(str(pair["workflow"]), str(pair["fault"]))].append(pair)
    slices["by_workflow_fault"] = [{"workflow": workflow, "fault": fault, **_summary(group, confidence)}
                                    for (workflow, fault), group in sorted(joint.items())]
    cross_kind = before_identity["agent_kind"] != after_identity["agent_kind"]
    return {
        "schema_version": "2.0", "before": before_identity, "after": after_identity,
        "comparison_kind": "cross_agent_kind" if cross_kind else "same_agent_kind",
        "cross_kind_note": "Scripted and model agents remain explicitly identified; this is not a same-model upgrade claim." if cross_kind else None,
        "decision": decision, "decision_reasons": reasons,
        "max_allowed_drop": float(max_allowed_drop), "max_allowed_drop_pp": 100.0 * max_allowed_drop,
        "summary": summary, "slices": slices,
        "safety_gate": {"status": "block" if safety_regressions else "pass", "regressions": safety_regressions,
                        "note": "This gate checks newly worse outcomes, not the absolute safety of either version."},
        "provider_errors": {"affected_pairs": len(provider_errors), "pairs": provider_errors,
                            "excluded_pairs": 0, "policy": "include_all_recorded_outcomes_and_prevent_pass"},
        "resources": {"cost_usd": _resource_delta(pairs, "cost_usd"),
                      "duration_seconds": _resource_delta(pairs, "duration_seconds")},
        "pairs": pairs,
        "limitations": ["A passing gate applies to this frozen suite and declared tolerance only.",
                        "Clustered intervals depend on a valid task-family definition and do not prove external validity.",
                        "Slice intervals are diagnostic; no multiple-comparison significance claim is made."],
    }
