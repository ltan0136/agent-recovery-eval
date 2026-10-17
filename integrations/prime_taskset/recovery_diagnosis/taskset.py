"""Native Verifiers 0.4.0 taskset for offline execution-trace diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

from recovery_diagnosis.scoring import build_prompt, exact_score, validate_diagnosis


class DiagnosisData(vf.TaskData):
    # References are evaluator data and are never included in the prompt.
    expected: dict[str, Any]


class DiagnosisTask(vf.Task[DiagnosisData]):
    @vf.reward
    async def diagnosis_exact(self, trace: vf.Trace) -> float:
        return exact_score(trace.last_reply, self.data.expected)


class DiagnosisConfig(vf.TasksetConfig):
    # An explicit path is preferred for runs launched outside this project folder.
    data_path: str = ""


def resolve_data_path(configured_path: str) -> Path:
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    packaged_path = Path(__file__).resolve().parent / "data" / "diagnosis_cases.jsonl"
    source_path = Path(__file__).resolve().parents[1] / "data" / "diagnosis_cases.jsonl"
    return packaged_path if packaged_path.is_file() else source_path


class DiagnosisTaskset(vf.Taskset[DiagnosisTask, DiagnosisConfig]):
    def load(self) -> list[DiagnosisTask]:
        data_path = resolve_data_path(self.config.data_path)
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Diagnosis cases are missing at {data_path}. Export core results "
                "first or pass --env.taskset.data-path /absolute/path/to/cases.jsonl"
            )
        tasks: list[DiagnosisTask] = []
        seen_ids: set[str] = set()
        with data_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Every case must be a JSON object")
                    case_id = row["id"]
                    if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
                        raise ValueError("Case IDs must be unique nonempty strings")
                    expected = validate_diagnosis(row["expected"])
                    prompt = build_prompt(row["request"], row["trace"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid case at {data_path}:{line_number}: {exc}") from exc
                seen_ids.add(case_id)
                tasks.append(
                    DiagnosisTask(
                        DiagnosisData(id=case_id, prompt=prompt, expected=expected),
                        self.config.task,
                    )
                )
        if not tasks:
            raise ValueError(f"No diagnosis cases found in {data_path}")
        return tasks
