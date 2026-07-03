"""Workload definitions for the benchmark suite.

Each workload is a reproducible definition of a real-world AI workload
pattern. The workloads are versioned so results can be compared across
versions of the suite.
"""

from .base import WorkloadSpec
from .registry import register_workload
from . import llm_decode, llm_prefill, cv_detection, audio_encoder, mixed_production

# Importing these modules registers their workloads via the @register decorator
# (or explicit register_workload calls in each module's __init__).

def all_workloads() -> list[WorkloadSpec]:
    """Return all registered workloads."""
    from .registry import list_workloads
    return list_workloads()
