"""Results aggregator.

Loads all JSON result files from the results/ directory and produces:
- A summary table (per workload, per agent)
- A leaderboard (ranked by avg_reward)
- A markdown report
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .runner import BenchmarkResult


@dataclass
class LeaderboardEntry:
    """A single entry in the leaderboard."""

    workload: str
    agent: str
    avg_reward: float
    total_time_ms: float
    total_energy_mj: float
    cache_hit_rate: float
    n_traces: int
    rank: int = 0

    def to_row(self) -> str:
        return (
            f"| {self.rank} | {self.workload} | {self.agent} | "
            f"{self.avg_reward:.4f} | {self.total_time_ms:.2f} | "
            f"{self.total_energy_mj:.2f} | {self.cache_hit_rate*100:.1f}% |"
        )


def load_results(results_dir: str = "benchmarks/results") -> list[BenchmarkResult]:
    """Load all JSON result files from the directory."""
    results: list[BenchmarkResult] = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return results

    for path in sorted(results_path.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            results.append(BenchmarkResult(**data))
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")

    return results


def aggregate_results(
    results: list[BenchmarkResult] | None = None,
    results_dir: str = "benchmarks/results",
) -> dict:
    """Aggregate results into a summary structure.

    Returns a dict with:
    - 'by_workload': {workload_name: [entries]}
    - 'by_agent': {agent_name: [entries]}
    - 'best_per_workload': {workload_name: best_entry}
    - 'leaderboard': [entries sorted by avg_reward desc]
    """
    if results is None:
        results = load_results(results_dir)

    by_workload: dict[str, list[BenchmarkResult]] = {}
    by_agent: dict[str, list[BenchmarkResult]] = {}

    for r in results:
        by_workload.setdefault(r.workload_name, []).append(r)
        by_agent.setdefault(r.agent_name, []).append(r)

    # Best per workload
    best_per_workload = {}
    for workload, entries in by_workload.items():
        best = max(entries, key=lambda e: e.avg_reward)
        best_per_workload[workload] = best

    # Full leaderboard (all results, sorted by avg_reward desc)
    leaderboard = sorted(results, key=lambda e: e.avg_reward, reverse=True)

    return {
        "by_workload": by_workload,
        "by_agent": by_agent,
        "best_per_workload": best_per_workload,
        "leaderboard": leaderboard,
    }


def generate_leaderboard(
    results: list[BenchmarkResult] | None = None,
    results_dir: str = "benchmarks/results",
) -> str:
    """Generate a markdown leaderboard from the results."""
    agg = aggregate_results(results, results_dir)

    lines = [
        "# Adaptive Firmware Layer — Benchmark Leaderboard",
        "",
        f"_Generated from {len(agg['leaderboard'])} benchmark runs_",
        "",
        "## Overall ranking (by avg_reward)",
        "",
        "| Rank | Workload | Agent | Avg Reward | Total Time (ms) | Total Energy (mJ) | Cache Hit |",
        "|------|----------|-------|-----------:|----------------:|------------------:|----------:|",
    ]

    for i, r in enumerate(agg["leaderboard"], start=1):
        entry = LeaderboardEntry(
            workload=r.workload_name,
            agent=r.agent_name,
            avg_reward=r.avg_reward,
            total_time_ms=r.total_time_ms,
            total_energy_mj=r.total_energy_mj,
            cache_hit_rate=r.cache_hit_rate,
            n_traces=r.n_traces,
            rank=i,
        )
        lines.append(entry.to_row())

    lines.extend([
        "",
        "## Best agent per workload",
        "",
        "| Workload | Best Agent | Avg Reward | Avg Time (ms) | Avg Energy (mJ) |",
        "|----------|-----------|-----------:|--------------:|----------------:|",
    ])

    for workload, r in sorted(agg["best_per_workload"].items()):
        lines.append(
            f"| {workload} | {r.agent_name} | {r.avg_reward:.4f} | "
            f"{r.total_time_ms:.2f} | {r.total_energy_mj:.2f} |"
        )

    lines.extend([
        "",
        "## Head-to-head: adaptive vs static",
        "",
        "For each workload, the adaptive agent's reward vs the best static config.",
        "",
        "| Workload | Adaptive | Best Static | Delta | Winner |",
        "|----------|---------:|-----------:|------:|--------|",
    ])

    for workload, entries in agg["by_workload"].items():
        adaptive_entries = [e for e in entries if e.agent_name in ("tabular", "neural")]
        static_entries = [e for e in entries if e.agent_name.startswith("static_")]

        if not adaptive_entries or not static_entries:
            continue

        best_adaptive = max(adaptive_entries, key=lambda e: e.avg_reward)
        best_static = max(static_entries, key=lambda e: e.avg_reward)
        delta = best_adaptive.avg_reward - best_static.avg_reward
        winner = "adaptive" if delta > 0 else "static"

        lines.append(
            f"| {workload} | {best_adaptive.avg_reward:.4f} ({best_adaptive.agent_name}) | "
            f"{best_static.avg_reward:.4f} ({best_static.agent_name}) | "
            f"{delta:+.4f} | {winner} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("To reproduce: `python -m benchmarks.runner`")
    lines.append("To submit your own results: open a PR with your JSON files in `benchmarks/results/`")

    return "\n".join(lines)
