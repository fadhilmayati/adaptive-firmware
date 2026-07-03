"""Random config baseline: pick a random config each step.

This baseline tests whether any agent is actually learning, or
if just being "different" each time is what helps. A random
agent should always lose to any reasonable adaptive agent.
"""

from __future__ import annotations

from ..runner import BenchmarkResult, BenchmarkRunner


class RandomConfigBaseline:
    """Run the workload with random config selection."""

    @staticmethod
    def run(workload_name: str, seed: int = 42) -> BenchmarkResult:
        runner = BenchmarkRunner()
        return runner.run(
            workload_name,
            agent="random",
            agent_config={"seed": seed},
        )
