"""Static config baseline: pick one config and never change it.

This is the simplest non-trivial baseline. If the workload is
homogeneous (all one workload class), a well-chosen static config
can be optimal or near-optimal. If the workload is heterogeneous,
a static config will leave performance on the table.
"""

from __future__ import annotations

from ..runner import BenchmarkResult, BenchmarkRunner
from adaptive_firmware.hardware.configs import CONFIG_PRESETS


class StaticConfigBaseline:
    """Run all four static configs and return the best one.

    In a real benchmark, you'd typically just pick one (or a few)
    configs to compare against. We run all four to find the best
    static for each workload — this is the upper bound on what
    "no adaptation" can achieve.
    """

    @staticmethod
    def run_all(workload_name: str) -> dict[int, BenchmarkResult]:
        """Run the workload with each static config."""
        runner = BenchmarkRunner()
        results = {}
        for config in CONFIG_PRESETS:
            agent = f"static_{config.config_id}"
            results[config.config_id] = runner.run(workload_name, agent=agent)
        return results

    @staticmethod
    def best(workload_name: str) -> BenchmarkResult:
        """Run all static configs and return the best one."""
        results = StaticConfigBaseline.run_all(workload_name)
        return max(results.values(), key=lambda r: r.avg_reward)
