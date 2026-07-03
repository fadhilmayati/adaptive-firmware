"""Standardized benchmark suite for adaptive reconfigurable hardware.

This suite provides reproducible workloads, a runner with JSON output,
and baseline implementations for comparison. Anyone working in this
space can run the same benchmarks and submit comparable results.

The suite is designed to be:
- Reproducible: all workloads are seeded
- Versioned: workloads have versions so results compare across versions
- Extensible: easy to add new workloads or baselines
- Comparable: standardized JSON output format
"""

from .runner import BenchmarkRunner, run_workload, run_all
from .workloads.base import WorkloadSpec
from .workloads.registry import list_workloads, get_workload, register_workload
from .aggregator import aggregate_results, generate_leaderboard

__version__ = "0.1.0"

__all__ = [
    "BenchmarkRunner",
    "run_workload",
    "run_all",
    "WorkloadSpec",
    "list_workloads",
    "get_workload",
    "register_workload",
    "aggregate_results",
    "generate_leaderboard",
]
