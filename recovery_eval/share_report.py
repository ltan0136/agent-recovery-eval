"""Render a standalone aggregate report using only run-wide numeric metrics."""

import json
from pathlib import Path

from .workflows import POLICIES


def _percentage(row: dict, field: str) -> str:
    value = row.get(field)
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError(f"Invalid aggregate rate: {field}")
    return f"{value:.1%}"


def export_summary(source: Path, destination: Path) -> None:
    summary = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), list):
        raise ValueError("Expected a summary.json with a metrics array")
    expected = summary.get("episodes")
    if type(expected) is not int or expected < 1:
        raise ValueError("Invalid total episode count")

    table_rows = []
    seen_agents = set()
    total = 0
    anonymous = 0
    for row in summary["metrics"]:
        if not isinstance(row, dict):
            raise ValueError("Expected an object for each metric")
        if row.get("workflow") != "all" or row.get("fault") != "all":
            continue
        agent = row.get("agent_id")
        if not isinstance(agent, str) or not agent or agent in seen_agents:
            raise ValueError("Expected one aggregate per distinct agent")
        seen_agents.add(agent)
        count, successes = row.get("episodes"), row.get("successes")
        if type(count) is not int or type(successes) is not int or not 0 <= successes <= count or count < 1:
            raise ValueError("Invalid aggregate episode or success count")
        if row.get("agent_kind") == "scripted" and agent in POLICIES:
            label = agent
        else:
            anonymous += 1
            label = f"匿名组 {anonymous}"
        duplicates = _percentage(row, "duplicate_episode_rate")
        missing = _percentage(row, "missing_episode_rate")
        false_success = _percentage(row, "false_success_rate")
        table_rows.append(
            f"<tr><th scope=\"row\">{label}</th><td>{count}</td><td>{successes}</td>"
            f"<td>{successes / count:.1%}</td><td>{duplicates}</td>"
            f"<td>{missing}</td><td>{false_success}</td></tr>"
        )
        total += count
    if total != expected:
        raise ValueError("Aggregate counts differ from the total number of episodes")

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>故障恢复评估汇总</title>
<style>
body {{font: 1rem/1.6 system-ui, sans-serif; color: #18283b; margin: 2rem auto; max-width: 68rem; padding: 0 1rem}}
.table-wrap {{overflow-x: auto}} table {{border-collapse: collapse; width: 100%; white-space: nowrap}}
th, td {{padding: .6rem; text-align: right; border-bottom: 1px solid #b2c2d0}}
th:first-child {{text-align: left}} thead {{background: #e7f0f7}}
</style></head><body><main><h1>故障恢复评估汇总</h1>
<p>共 {total} 个回合。每行汇总一个评测组在所有工作流和故障条件下的结果。</p>
<div class="table-wrap"><table><thead><tr><th>评测组</th><th>回合数</th><th>成功数</th>
<th>成功率</th><th>重复执行回合占比</th><th>漏做回合占比</th><th>误报成功回合占比</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</main></body></html>
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(page)
