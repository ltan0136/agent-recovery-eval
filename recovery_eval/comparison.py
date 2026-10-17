"""Paired regression detection on matching, frozen evaluation conditions."""

from __future__ import annotations

import html
import json
from pathlib import Path


def load_rows(path: str | Path, policy: str) -> dict[str, dict]:
    selected = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["policy"] == policy:
            if row["case_id"] in selected:
                raise ValueError("Duplicate case IDs within a policy; compare distinct repeated trials explicitly")
            selected[row["case_id"]] = row
    if not selected:
        raise ValueError(f"No episodes for policy {policy}")
    return selected


def compare_files(before: str, after: str, before_policy: str, after_policy: str,
                  out: str = "results/regression") -> dict:
    left, right = load_rows(before, before_policy), load_rows(after, after_policy)
    if set(left) != set(right):
        raise ValueError("Case sets differ; refusing an unpaired comparison")
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if not a.get("condition_hash") or a["condition_hash"] != b.get("condition_hash"):
            raise ValueError(f"Frozen conditions differ or are missing: {key}")
        label = ("regression" if a["success"] and not b["success"] else
                 "improvement" if not a["success"] and b["success"] else "unchanged")
        pairs.append({"case_id": key, "change": label,
                      "before_success": a["success"], "after_success": b["success"],
                      "before_duplicates": a["duplicates"], "after_duplicates": b["duplicates"],
                      "tool_call_delta": b["tool_calls"] - a["tool_calls"]})
    summary = {"before_policy": before_policy, "after_policy": after_policy,
               "paired_cases": len(pairs),
               "regressions": sum(p["change"] == "regression" for p in pairs),
               "improvements": sum(p["change"] == "improvement" for p in pairs),
               "unchanged": sum(p["change"] == "unchanged" for p in pairs),
               "note": "Descriptive paired comparison on this frozen suite, not a population-level significance claim.",
               "pairs": pairs}
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    changed = [p for p in pairs if p["change"] != "unchanged"]
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(p[k]))}</td>" for k in
                    ("case_id", "change", "before_success", "after_success", "after_duplicates")) + "</tr>"
                   for p in changed)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>版本回归对照</title><style>body{{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#183047}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border:1px solid #d7e1e8}}th{{background:#edf3f7}}</style>
<h1>配对版本回归对照</h1><p>{html.escape(before_policy)} → {html.escape(after_policy)}</p>
<p>同条件配对 {len(pairs)} 回合；退化 {summary['regressions']}；改善 {summary['improvements']}；无变化 {summary['unchanged']}。</p>
<p>每个案例均核对完整条件哈希；以下只是这组固定仿真案例的描述结果，不代表真实模型或跨业务总体结论。</p>
<table><tr><th>案例</th><th>变化</th><th>旧版完成</th><th>新版完成</th><th>新版重复数</th></tr>{rows}</table>
<p><a href="comparison.json">完整对照 JSON</a></p></html>"""
    (directory / "report.html").write_text(page, encoding="utf-8")
    return summary
