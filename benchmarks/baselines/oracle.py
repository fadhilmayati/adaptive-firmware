"""Oracle baseline: always pick the best config for the workload class.

The oracle knows the optimal mapping (workload_class → best config)
and uses it perfectly. This is the upper bound on what any adaptive
agent can achieve.

The oracle is intentionally not realistic — in production you don't
know the optimal mapping ahead of time. The point is to measure
how much room there is to improve, and how close adaptive agents
get to the ceiling.
"""

from __future__ import annotations

from ..runner import BenchmarkResult, BenchmarkRunner


class OracleBaseline:
    """Run the workload with the optimal config for each workload class."""

    @staticmethod
    def run(workload_name: str) -> BenchmarkResult:
        runner = BenchmarkRunner()
        return runner.run(workload_name, agent="oracle")
