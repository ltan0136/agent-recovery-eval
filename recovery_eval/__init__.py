"""Multi-workflow tool execution and recovery evaluation.

The v0.1 two-step API remains available in recovery_eval.core; current callers
should use Environment, build_matrix and run_scripted.
"""

from .engine import Environment, run_scripted
from .workflows import FAULTS, POLICIES, build_matrix, workflow_specs

__version__ = "0.2.0"
__all__ = ["Environment", "FAULTS", "POLICIES", "build_matrix", "run_scripted", "workflow_specs"]
