"""Export the native Verifiers v1 Taskset and a local convenience loader."""

from recovery_diagnosis.taskset import DiagnosisConfig, DiagnosisTaskset


def load_environment(data_path: str = "") -> DiagnosisTaskset:
    """Return a v1 Taskset; this helper does not start a runtime or model call."""
    return DiagnosisTaskset(DiagnosisConfig(data_path=data_path))


__all__ = ["DiagnosisConfig", "DiagnosisTaskset", "load_environment"]
