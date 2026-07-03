"""Base class for standardized workload definitions.

A WorkloadSpec is a reproducible definition of a workload that can be
run by the benchmark suite. It includes:
- A versioned name (so results can be compared across versions)
- A description (what the workload represents)
- Tags (for filtering and categorization)
- A trace generator function (deterministic given the seed)
- Expected metadata (n_traces, workload class distribution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from adaptive_firmware.observation.telemetry import WorkloadTrace


@dataclass(frozen=True)
class WorkloadSpec:
    """A standardized, reproducible workload definition.

    Attributes:
        name: Short identifier (e.g., "llm_decode").
        version: Semver string (e.g., "1.0.0"). Bump when the workload
                 definition changes so old results can be invalidated.
        description: One-line description of what this workload represents.
        tags: List of tags for filtering (e.g., ["llm", "memory_bound"]).
        workload_class: "single-tenant" or "multi-tenant".
        trace_generator: Callable that returns a list of WorkloadTrace.
                        Must be deterministic given the same RNG state.
        expected_n_traces: Expected number of traces (for validation).
        seed: Random seed for this workload (default 42).
    """

    name: str
    version: str
    description: str
    tags: list[str]
    workload_class: str
    trace_generator: Callable[[int], list[WorkloadTrace]]
    expected_n_traces: int
    seed: int = 42

    def generate(self) -> list[WorkloadTrace]:
        """Generate the workload traces (deterministic)."""
        return self.trace_generator(self.seed)

    def validate(self, traces: list[WorkloadTrace], tolerance: float = 0.0) -> bool:
        """Check that the generated traces match the expected shape.

        Args:
            traces: The generated traces to validate.
            tolerance: Fractional tolerance for length mismatch (0 = exact).
                      Useful for workloads with random element counts.
        """
        expected = self.expected_n_traces
        if expected > 0:
            if tolerance > 0:
                lo = int(expected * (1 - tolerance))
                hi = int(expected * (1 + tolerance))
                if not (lo <= len(traces) <= hi):
                    return False
            elif len(traces) != expected:
                return False
        if not all(t.flops > 0 for t in traces):
            return False
        if not all(t.memory_bytes >= 0 for t in traces):
            return False
        return True
