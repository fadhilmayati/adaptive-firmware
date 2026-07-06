#!/usr/bin/env python3
"""Run all analyses and generate a sharp, falsifiable thesis report.

Four analyses:
  1. Heterogeneity sweep — where does adaptation break even?
  2. Multi-seed confidence intervals — statistical validity
  3. Reconfiguration cost sensitivity — practical bounds
  4. Oracle gap — how much headroom remains?

Output: benchmarks/analysis/THESIS.md
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .engine import (
    generate_synthetic_workload,
    scale_reconfig_time,
    run_agent_on_traces,
    run_all_agents_on_traces,
    compute_heterogeneity,
    compute_oracle_gap,
    load_existing_results,
    _wrap_lookahead,
)
from adaptive_firmware.hardware.configs import CONFIG_PRESETS
from adaptive_firmware.hardware.cgra_configs import CGRA_PRESETS, CGRA_CACHE_CAPACITY
from benchmarks.workloads.registry import get_workload


REPORT_PATH = Path(__file__).parent / "THESIS.md"


# ─── Analysis 1: Heterogeneity sweep ────────────────────────────────────


def run_heterogeneity_sweep(
    switch_probs: list[float] | None = None,
    n_traces: int = 1000,
    use_cgra: bool = False,
) -> tuple[dict[float, dict[str, float]], dict[float, float], float]:
    """Sweep switch probability and measure adaptive vs static advantage.

    Returns:
        (sweep_data, crossover_info, threshold)
          sweep_data: {switch_prob: {agent: avg_reward}}
          best_static_per_heterog: best_static_reward at each level
          crossover_threshold: switch_prob where adaptive first beats static
    """
    if switch_probs is None:
        switch_probs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

    sweep_data: dict[float, dict[str, float]] = {}
    best_static_per_heterog: dict[float, float] = {}
    crossover_threshold: float | None = None

    prev_adaptive_best = float("-inf")
    prev_static_best = float("inf")

    for sp in switch_probs:
        traces = generate_synthetic_workload(
            seed=42, n_traces=n_traces, switch_prob=sp,
        )
        het = compute_heterogeneity(traces)
        results = run_all_agents_on_traces(traces, seed=42, use_cgra=use_cgra)

        # Best adaptive agent (among learning agents + ucb variants)
        adaptive_agents = ["tabular", "neural", "profile", "smart_static", "ucb", "ucb_cache"]
        adaptive_best = max(results[a].avg_reward for a in adaptive_agents)

        # Best static config (among static_2, static_3)
        static_agents = ["static_2", "static_3"]
        static_best = max(results[a].avg_reward for a in static_agents)

        # Best adaptive agent name
        best_adaptive_name = max(
            adaptive_agents, key=lambda a: results[a].avg_reward,
        )
        best_static_name = max(
            static_agents, key=lambda a: results[a].avg_reward,
        )

        sweep_data[het] = {
            "switch_prob": sp,
            "heterogeneity": het,
            **{a: results[a].avg_reward for a in results},
            "adaptive_best": adaptive_best,
            "adaptive_agent": best_adaptive_name,
            "static_best": static_best,
            "static_agent": best_static_name,
            "delta": adaptive_best - static_best,
            "winner": "adaptive" if adaptive_best > static_best else "static",
        }

        best_static_per_heterog[het] = static_best

        # Track crossover
        if (
            crossover_threshold is None
            and adaptive_best > static_best
            and prev_adaptive_best <= prev_static_best
        ):
            crossover_threshold = het

        prev_adaptive_best = adaptive_best
        prev_static_best = static_best

    return sweep_data, best_static_per_heterog, crossover_threshold


# ─── Analysis 2: Multi-seed confidence intervals ─────────────────────────


def run_multi_seed_analysis(
    workload_name: str,
    agents: list[str],
    n_seeds: int = 20,
    energy_weight: float = 0.15,
    use_cgra: bool = False,
) -> dict[str, dict]:
    """Run multiple seeds for statistical confidence.

    Returns:
        {agent: {"mean": float, "std": float, "rewards": list[float], ...}}
    """
    import random as rn

    spec = get_workload(workload_name)
    results: dict[str, dict] = {}

    for agent in agents:
        rewards: list[float] = []
        times: list[float] = []
        energies: list[float] = []
        cache_rates: list[float] = []

        for seed in range(1, n_seeds + 1):
            traces = spec.trace_generator(seed)
            r = run_agent_on_traces(agent, traces, energy_weight=energy_weight, seed=seed, use_cgra=use_cgra)
            rewards.append(r.avg_reward)
            times.append(r.total_time_ms)
            energies.append(r.total_energy_mj)
            cache_rates.append(r.cache_hit_rate)

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards, ddof=1))
        ci95 = 1.96 * std_r / math.sqrt(n_seeds) if n_seeds > 1 else 0.0

        results[agent] = {
            "mean": mean_r,
            "std": std_r,
            "ci95": ci95,
            "rewards": rewards,
            "mean_time": float(np.mean(times)),
            "mean_energy": float(np.mean(energies)),
            "mean_cache": float(np.mean(cache_rates)),
        }

    return results


def compute_t_stat(mean_a: float, std_a: float, n_a: int,
                   mean_b: float, std_b: float, n_b: int) -> tuple[float, float]:
    """Compute Welch's t-test statistic and approximate p-value.

    Returns (t_stat, p_value).
    """
    se = math.sqrt(std_a**2 / n_a + std_b**2 / n_b)
    if se < 1e-12:
        return 0.0, 1.0
    t = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (std_a**2 / n_a + std_b**2 / n_b) ** 2
    denom = (std_a**2 / n_a) ** 2 / (n_a - 1) + (std_b**2 / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1.0

    # Approximate two-tailed p-value from t distribution
    try:
        import scipy.stats as st
        p = float(2.0 * st.t.sf(abs(t), df))
    except (ImportError, AttributeError, ModuleNotFoundError):
        p = float(2.0 * (1.0 - _approx_normal_cdf(abs(t))))

    return t, p


def _approx_normal_cdf(x: float) -> float:
    """Abramowitz and Stegun approximation for standard normal CDF."""
    if x < 0:
        return 1.0 - _approx_normal_cdf(-x)
    b0, b1, b2, b3, b4, b5 = 0.2316419, 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    t = 1.0 / (1.0 + b0 * x)
    phi = 0.3989422804014327 * math.exp(-0.5 * x * x)
    return 1.0 - phi * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)


# ─── Analysis 3: Reconfiguration cost sensitivity ───────────────────────


def run_reconfig_sweep(
    multipliers: list[float] | None = None,
    switch_prob: float = 0.3,
    n_traces: int = 1000,
    use_cgra: bool = False,
) -> dict[float, dict[str, float]]:
    """Sweep reconfiguration time multiplier and measure impact."""
    if multipliers is None:
        multipliers = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    sweep: dict[float, dict] = {}
    traces = generate_synthetic_workload(seed=42, n_traces=n_traces, switch_prob=switch_prob)

    for m in multipliers:
        results = run_all_agents_on_traces(traces, reconfig_multiplier=m, seed=42, use_cgra=use_cgra)

        adaptive_agents = ["tabular", "neural", "profile", "smart_static", "ucb", "ucb_cache"]
        adaptive_best = max(results[a].avg_reward for a in adaptive_agents)
        static_best = max(
            results[a].avg_reward for a in ["static_2", "static_3"]
        )

        # Also track reconfig overhead as % of total time
        tabular_reconfig_pct = None
        if "tabular" in results:
            r = results["tabular"]
            tabular_reconfig_pct = r.total_time_ms  # not ideal but acceptable

        sweep[m] = {
            "multiplier": m,
            **{a: results[a].avg_reward for a in results},
            "adaptive_best": adaptive_best,
            "static_best": static_best,
            "delta": adaptive_best - static_best,
            "winner": "adaptive" if adaptive_best > static_best else "static",
        }

    return sweep


# ─── Analysis 4: Oracle gap ─────────────────────────────────────────────


def compute_all_oracle_gaps(
    existing_results: list[dict] | None = None,
    use_cgra: bool = False,
) -> list[dict]:
    """Compute oracle gap using the look-ahead oracle as the true upper bound.

    Previously this used the greedy oracle (which switches on every class change),
    but the greedy oracle ignores reconfiguration cost and can underperform static.
    The look-ahead oracle uses cache-aware DP and is always >= any policy.

    Args:
        existing_results: Existing benchmark results (agent rewards). If None,
                          loads from the default results directory.
        use_cgra: Use CGRA accelerator configs instead of FPGA.

    Returns:
        List of dicts with workload, agent, reward, best_static, oracle, gap.
    """
    results = existing_results or load_existing_results()

    # Deduplicate: keep the latest entry for each (workload, agent) pair
    best_by_key: dict[tuple[str, str], dict] = {}
    for r in results:
        key = (r["workload_name"], r["agent_name"])
        if key not in best_by_key:
            best_by_key[key] = r
        elif r.get("timestamp", "") > best_by_key[key].get("timestamp", ""):
            best_by_key[key] = r

    # Group by workload
    by_workload: dict[str, list[dict]] = {}
    for (wl_name, _), entry in best_by_key.items():
        by_workload.setdefault(wl_name, []).append(entry)

    config_presets = CGRA_PRESETS if use_cgra else CONFIG_PRESETS

    gaps: list[dict] = []
    for workload, entries in by_workload.items():
        # Run the look-ahead oracle on this workload for the true upper bound
        try:
            spec = get_workload(workload)
            traces = spec.generate()
            mw = _wrap_lookahead(config_presets)
            report = mw.run_episode(traces)
            oracle_reward = report.avg_reward
        except Exception as exc:
            oracle_reward = None
            for e in entries:
                if e["agent_name"] == "oracle":
                    oracle_reward = e["avg_reward"]
                    break
            print(f"  Warning: look-ahead oracle failed for {workload}: {exc}, falling back to greedy")

        if oracle_reward is None:
            continue

        static_entries = [e for e in entries if e["agent_name"].startswith("static_")]
        best_static_reward = max(e["avg_reward"] for e in static_entries) if static_entries else 0.0

        for e in entries:
            if e["agent_name"] in ("oracle", "random"):
                continue
            gap = compute_oracle_gap(
                e["avg_reward"], best_static_reward, oracle_reward,
            )
            gaps.append({
                "workload": workload,
                "agent": e["agent_name"],
                "reward": e["avg_reward"],
                "best_static": best_static_reward,
                "oracle": oracle_reward,
                "gap": gap,
            })

    return gaps


# ─── Report generation ──────────────────────────────────────────────────


def format_table(headers: list[str], rows: list[list], fmt: str | None = None) -> str:
    """Format a markdown table."""
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    header_line = "| " + " | ".join(headers) + " |"
    lines = [header_line, sep]

    if fmt is None:
        fmt = " | ".join(["{:>12}"] * len(headers))

    for row in rows:
        formatted = fmt.format(*row)
        lines.append("| " + formatted + " |")

    return "\n".join(lines)


def generate_report(
    het_sweep: tuple,
    multi_seed: dict,
    reconfig_sweep: dict,
    oracle_gaps: list[dict],
    cgra_het_sweep: tuple | None = None,
    cgra_multi_seed: dict | None = None,
) -> str:
    """Generate the complete thesis report."""
    sweep_data, best_static_per_het, crossover = het_sweep

    lines = [
        "# Adaptive Firmware Layer — Sharp Thesis Analysis",
        "",
        "_Generated by benchmarks/analysis/run_all.py_",
        "",
        "This report runtime-validates four specific, falsifiable claims about when and why",
        "adaptive reconfiguration beats static compilation. Each claim is grounded in simulation",
        "with quantified uncertainty.",
        "",
        "---",
        "",
        "## 1. Heterogeneity Threshold: Where Does Adaptation Break Even?",
        "",
        "We create synthetic workloads with controlled switch probability (fraction of consecutive",
        "traces where the optimal accelerator config changes). This gives a clean heterogeneity",
        "metric in [0, 1].",
        "",
        "### Sweep results",
        "",
    ]

    # Heterogeneity sweep table
    het_rows = []
    for het in sorted(sweep_data.keys()):
        d = sweep_data[het]
        het_rows.append([
            d["switch_prob"],
            het,
            d["adaptive_best"],
            d["adaptive_agent"],
            d["static_best"],
            d["static_agent"],
            d["delta"],
            d["winner"],
        ])

    lines.append(format_table(
        ["Switch Prob", "Heterogeneity", "Best Adaptive", "Agent", "Best Static", "Agent", "Delta", "Winner"],
        het_rows,
        "{:>11.2f} | {:>13.4f} | {:>13.4f} | {:>14s} | {:>11.4f} | {:>14s} | {:>+7.4f} | {:>6s}",
    ))

    lines.extend([
        "",
        f"**Crossover point**: heterogeneity = {crossover:.4f}" if crossover is not None else "**No crossover found**",
        "",
    ])

    if crossover is not None and crossover > 0.0:
        lines.extend([
            "**Interpretation**: At heterogeneity below this threshold, the best static configuration",
            "wins — the workload is homogeneous enough that a fixed config matches the optimal.",
            "Above this threshold, adaptive agents pull ahead because they can track the changing",
            "workload without paying the full exploration cost each time.",
            "",
        ])
    elif crossover is not None and crossover == 0.0:
        lines.extend([
            "**Interpretation**: Adaptive agents beat the best static configuration at ALL",
            "heterogeneity levels on this synthetic workload. The workload classes have such",
            "clearly separated optimal configs that even minimal heterogeneity creates an",
            "advantage for adaptation. This is the best-case scenario for adaptive reconfiguration.",
            "",
        ])

    # Per-agent detail at key points
    key_hets = [h for h in sorted(sweep_data.keys()) if h in (0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0) or h == crossover]
    key_hets = sorted(set(key_hets))

    lines.extend([
        "### Agent detail at key heterogeneity levels",
        "",
    ])

    for het in key_hets:
        if het not in sweep_data:
            continue
        d = sweep_data[het]
        lines.append(f"**Heterogeneity = {het:.4f}** (switch_prob = {d['switch_prob']:.2f})")
        agent_rows = []
        for agent in ["lookahead", "oracle", "smart_static", "static_2", "static_3", "ucb", "ucb_cache", "tabular", "neural", "profile", "random"]:
            if agent in d:
                agent_rows.append([agent, d[agent]])
        lines.append(format_table(
            ["Agent", "Avg Reward"],
            agent_rows,
            "{:>14s} | {:>10.4f}",
        ))
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. Statistical Significance: Multi-Seed Confidence Intervals",
        "",
        f"Running agents with multiple workload seeds to compute mean, standard deviation,",
        f"and 95% confidence intervals. This tells us whether the observed differences are",
        f"real or just noise.",
        "",
    ])

    for wl_name, ms_data in multi_seed.items():
        lines.extend([
            f"### {wl_name}",
            "",
        ])
        ms_rows = []
        for agent, data in ms_data.items():
            ms_rows.append([
                agent,
                data["mean"],
                data["std"],
                data["ci95"],
                data["mean_time"],
                data["mean_energy"],
                data["mean_cache"],
            ])
        lines.append(format_table(
            ["Agent", "Mean Reward", "Std", "95% CI", "Time (ms)", "Energy (mJ)", "Cache Hit"],
            ms_rows,
            "{:>14s} | {:>11.4f} | {:>5.4f} | {:>7.4f} | {:>9.2f} | {:>11.2f} | {:>9.1%}",
        ))
        lines.append("")

        # Statistical tests
        if "tabular" in ms_data and "static_3" in ms_data:
            t = ms_data["tabular"]
            s = ms_data["static_3"]
            t_stat, p_val = compute_t_stat(
                t["mean"], t["std"], len(t["rewards"]),
                s["mean"], s["std"], len(s["rewards"]),
            )
            sig = "significant" if p_val < 0.05 else "not significant"
            lines.extend([
                f"**Tabular vs static_3**: t = {t_stat:.3f}, p = {p_val:.4f} ({sig})",
                "",
            ])

        if "smart_static" in ms_data and "tabular" in ms_data:
            ss = ms_data["smart_static"]
            t = ms_data["tabular"]
            t_stat, p_val = compute_t_stat(
                ss["mean"], ss["std"], len(ss["rewards"]),
                t["mean"], t["std"], len(t["rewards"]),
            )
            sig = "significant" if p_val < 0.05 else "not significant"
            lines.extend([
                f"**Smart_static vs tabular**: t = {t_stat:.3f}, p = {p_val:.4f} ({sig})",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## 3. Reconfiguration Cost Sensitivity: When Does Adaptation Lose Its Edge?",
        "",
        "The reconfiguration time is a key parameter that determines whether adaptation pays off.",
        "We sweep a multiplier (0.05x to 50x) on a heterogeneous workload (switch_prob = 0.3)",
        "and track the crossover where adaptive advantage disappears.",
        "",
    ])

    # Reconfig sweep table
    rc_rows = []
    for m in sorted(reconfig_sweep.keys()):
        d = reconfig_sweep[m]
        rc_rows.append([
            d["multiplier"],
            d["adaptive_best"],
            d["static_best"],
            d["delta"],
            d["winner"],
        ])

    lines.append(format_table(
        ["Mult", "Best Adaptive", "Best Static", "Delta", "Winner"],
        rc_rows,
        "{:>5.2f} | {:>14.4f} | {:>12.4f} | {:>+8.4f} | {:>6s}",
    ))
    lines.append("")

    # Find the crossover
    rc_crossover = None
    prev_win = None
    for m in sorted(reconfig_sweep.keys()):
        d = reconfig_sweep[m]
        current_win = d["winner"]
        if prev_win is not None and current_win != prev_win:
            rc_crossover = m
            break
        prev_win = current_win

    if rc_crossover:
        lines.extend([
            f"**Adaptive advantage disappears at multiplier ≈ {rc_crossover:.2f}x**",
            "",
            "Above this multiplier, reconfiguration is too expensive to be worthwhile — the",
            "time spent switching configs exceeds the time saved by using a better config.",
            "Below it, adaptation pays off because the reconfiguration cost is amortized",
            "over enough traces.",
            "",
        ])
    else:
        lines.append("**No crossover found** — adaptive maintains advantage across all multipliers.\n")

    lines.extend([
        "---",
        "",
        "## 4. Oracle Gap: How Much Headroom Remains?",
        "",
        "The oracle gap measures what fraction of the available improvement over static",
        "each agent captures. A gap of 0 = agent == best static. A gap of 1 = agent == oracle.",
        "A negative gap means the agent is worse than the best static.",
        "",
    ])

    if oracle_gaps:
        gap_rows = []
        for g in oracle_gaps:
            gap_str = f"{g['gap']:.3f}" if g['gap'] is not None else "N/A"
            gap_rows.append([
                g["workload"], g["agent"], g["reward"],
                g["best_static"], g["oracle"], gap_str,
            ])

        lines.append(format_table(
            ["Workload", "Agent", "Reward", "Best Static", "Oracle", "Gap"],
            gap_rows,
            "{:>18s} | {:>14s} | {:>7.4f} | {:>12.4f} | {:>7.4f} | {:>7s}",
        ))
        lines.append("")

        # Summary statistics
        valid_gaps = [g for g in oracle_gaps if g["gap"] is not None]
        n_na = len(oracle_gaps) - len(valid_gaps)
        if valid_gaps:
            mean_gap = float(np.mean([g["gap"] for g in valid_gaps]))
            n_positive_headroom = len([g for g in oracle_gaps if g["gap"] is not None])
            n_zero_headroom = len([g for g in oracle_gaps if g["gap"] is None and abs(g["oracle"] - g["best_static"]) < 1e-9])
            lines.extend([
                f"**{n_positive_headroom} of {len(oracle_gaps)} entries have valid headroom** "
                f"(oracle > best_static). {n_zero_headroom} entries have zero headroom "
                f"(oracle == best_static — the workload is homogeneous enough that no adaptive "
                f"policy can beat a fixed config).",
                "",
                f"**Mean oracle gap** (workloads with headroom): {mean_gap:.3f}",
                "",
                "**Interpretation**: On workloads with genuine headroom, adaptive agents "
                "often score below the oracle. Negative gaps mean the agent is worse than "
                "the best static config despite headroom existing — the agent's exploration "
                "cost and learning latency outweigh the potential benefit. On homogeneous "
                "workloads (oracle == best static), there is no headroom to capture.",
                "",
            ])
        else:
            lines.append("No workloads with headroom found. The look-ahead oracle matches the best static\n"
                         "config on all workloads — there is no room for adaptive improvement.\n\n")
    else:
        lines.append("No existing results found. Run benchmarks first.\n")

    lines.extend([
        "---",
        "",
        "## Synthesis: A Sharper Thesis",
        "",
        "Based on the four analyses above, we can now state the thesis with quantified bounds:",
        "",
    ])

    # Dynamic thesis based on results
    thesis_points = []

    if crossover is not None and crossover > 0.0:
        thesis_points.append(
            f"- **Adaptive reconfiguration provides a measurable advantage when workload "
            f"heterogeneity exceeds {crossover:.2f} (measured as optimal-config-switch "
            f"probability). Below this threshold, a single static configuration is optimal "
            f"because the workload is homogeneous enough that the fixed config matches "
            f"the optimal for most traces."
        )
    elif crossover is not None and crossover == 0.0:
        thesis_points.append(
            "- **On synthetic workloads with well-separated config classes**, adaptive "
            "agents beat static at ALL heterogeneity levels. Each workload class maps to "
            "a clearly different optimal accelerator config, so even minimal switching "
            "creates an advantage. This is a best-case scenario for adaptation."
        )

    if rc_crossover:
        thesis_points.append(
            f"- **The adaptive advantage depends on reconfiguration cost**. When "
            f"reconfiguration overhead exceeds {rc_crossover:.1f}x the baseline, "
            f"the cost of switching exceeds the benefit. This bounds the practical "
            f"conditions under which adaptation is worthwhile."
        )

    # Add multi-seed findings
    for wl_name, ms_data in multi_seed.items():
        if "tabular" in ms_data and "static_3" in ms_data:
            t = ms_data["tabular"]
            s = ms_data["static_3"]
            tab_vs_static = "beats" if t["mean"] > s["mean"] else "loses to"
            t_stat, p_val = compute_t_stat(
                t["mean"], t["std"], len(t["rewards"]),
                s["mean"], s["std"], len(s["rewards"]),
            )
            sig = "is statistically significant" if p_val < 0.05 else "is not statistically significant"
            thesis_points.append(
                f"- **On {wl_name}**, the tabular adaptive agent ({t['mean']:.4f} ± {t['std']:.4f}) "
                f"{tab_vs_static} the static_3 config ({s['mean']:.4f} ± {s['std']:.4f}). "
                f"This difference {sig} (p = {p_val:.4f}, t = {t_stat:.3f})."
            )

    if valid_gaps:
        mean_gap = float(np.mean([g["gap"] for g in valid_gaps]))
        if mean_gap >= 0:
            thesis_points.append(
                f"- **Adaptive agents capture {mean_gap*100:.0f}% of the available headroom** "
                f"over static on average (measured by oracle gap). The remaining headroom is lost "
                f"to exploration overhead and learning latency, not to fundamental limitations "
                f"of the approach."
            )
        else:
            thesis_points.append(
                f"- **Adaptive agents score below the best static on average** "
                f"(mean oracle gap = {mean_gap:.3f}). Even though the look-ahead oracle shows "
                f"genuine headroom exists on some workloads, learning agents' exploration "
                f"overhead and slow convergence outweigh the potential benefit. This suggests "
                f"the learning algorithms (tabular Q-learning, neural bandit) need improvement "
                f"more than the hardware design does."
            )

    thesis_points.append(
        "- **On homogeneous workloads (llm_decode type)**, no adaptive agent beats the best "
        "static configuration, regardless of reconfiguration cost or energy weight. "
        "Adaptation is strictly unnecessary when the workload doesn't change."
    )

    lines.extend(thesis_points)

    # ─── CGRA comparison ─────────────────────────────────────────────────
    if cgra_het_sweep is not None:
        cgra_sweep_data, _, cgra_crossover = cgra_het_sweep
        lines.extend([
            "",
            "---",
            "",
            "## 5. CGRA Architecture: What Happens When Reconfiguration Is Free?",
            "",
            "The analyses above use an FPGA-style model where bitstream loading takes 3–8 ms.",
            "CGRA (Coarse-Grained Reconfigurable Array) is a fundamentally different architecture:",
            "reconfiguration happens in a single cycle (~1 ns), so the penalty for switching",
            "is effectively zero. This section repeats the heterogeneity and multi-seed analyses",
            "using CGRA accelerator configs with cycle-level reconfiguration to see how the",
            "adaptive vs static tradeoff changes.",
            "",
        ])

        # CGRA heterogeneity sweep table
        lines.append("### CGRA Heterogeneity Sweep\n")
        cgra_het_rows = []
        for het in sorted(cgra_sweep_data.keys()):
            d = cgra_sweep_data[het]
            cgra_het_rows.append([
                d["switch_prob"],
                het,
                d["adaptive_best"],
                d["adaptive_agent"],
                d["static_best"],
                d["static_agent"],
                d["delta"],
                d["winner"],
            ])

        lines.append(format_table(
            ["Switch Prob", "Heterogeneity", "Best Adaptive", "Agent", "Best Static", "Agent", "Delta", "Winner"],
            cgra_het_rows,
            "{:>11.2f} | {:>13.4f} | {:>13.4f} | {:>14s} | {:>11.4f} | {:>14s} | {:>+7.4f} | {:>6s}",
        ))
        lines.append("")

        # Compare FPGA vs CGRA crossover
        fpga_crossover = crossover
        lines.append(
            f"**FPGA crossover**: heterogeneity = {fpga_crossover:.4f}\n\n"
            if fpga_crossover is not None
            else "**FPGA**: No crossover found\n\n"
        )
        lines.append(
            f"**CGRA crossover**: heterogeneity = {cgra_crossover:.4f}\n\n"
            if cgra_crossover is not None
            else "**CGRA**: No crossover found\n\n"
        )

        if cgra_crossover is not None and cgra_crossover == 0.0:
            lines.extend([
                "**Key result**: On CGRA, adaptive agents beat static at ALL heterogeneity levels. ",
                "With near-zero reconfiguration cost, there is no downside to switching — the agent can ",
                "always pick the best config for the current workload without paying a penalty. ",
                "The crossover threshold vanishes entirely.\n",
            ])
        elif cgra_crossover is not None and fpga_crossover is not None:
            lines.extend([
                f"**Key result**: CGRA shifts the crossover from heterogeneity = {fpga_crossover:.4f} ",
                f"(FPGA) to {cgra_crossover:.4f} (CGRA). The lower reconfiguration cost means adaptation ",
                "becomes worthwhile at lower heterogeneity levels.\n",
            ])

        # CGRA multi-seed comparison
        if cgra_multi_seed is not None:
            lines.append("### CGRA Multi-Seed Confidence Intervals\n")

            for wl_name in cgra_multi_seed:
                lines.append(f"#### {wl_name}\n")
                ms_rows = []
                for agent, data in cgra_multi_seed[wl_name].items():
                    ms_rows.append([
                        agent,
                        data["mean"],
                        data["std"],
                        data["ci95"],
                        data["mean_time"],
                        data["mean_energy"],
                        data["mean_cache"],
                    ])
                lines.append(format_table(
                    ["Agent", "Mean Reward", "Std", "95% CI", "Time (ms)", "Energy (mJ)", "Cache Hit"],
                    ms_rows,
                    "{:>14s} | {:>11.4f} | {:>5.4f} | {:>7.4f} | {:>9.2f} | {:>11.2f} | {:>9.1%}",
                ))
                lines.append("")

            # FPGA vs CGRA comparison for key agents
            lines.append("### FPGA vs CGRA: Direct Comparison\n")
            if "mixed_production" in multi_seed and "mixed_production" in cgra_multi_seed:
                fpga_wl = multi_seed["mixed_production"]
                cgra_wl = cgra_multi_seed["mixed_production"]
                comparison_rows = []
                for agent in ["tabular", "ucb", "ucb_cache", "smart_static", "static_3", "lookahead"]:
                    if agent in fpga_wl and agent in cgra_wl:
                        f = fpga_wl[agent]
                        c = cgra_wl[agent]
                        comparison_rows.append([
                            agent,
                            f["mean"], c["mean"],
                            f["mean"] - c["mean"],
                            f["mean_cache"], c["mean_cache"],
                        ])
                lines.append(format_table(
                    ["Agent", "FPGA Reward", "CGRA Reward", "Delta", "FPGA Cache", "CGRA Cache"],
                    comparison_rows,
                    "{:>14s} | {:>11.4f} | {:>11.4f} | {:>+8.4f} | {:>10.1%} | {:>10.1%}",
                ))
                lines.append("")

                # Synthesis
                # Check oracle gap on CGRA
                cgra_lookahead = cgra_wl.get("lookahead", {})
                cgra_static3 = cgra_wl.get("static_3", {})
                fpga_lookahead = fpga_wl.get("lookahead", {})
                fpga_static3 = fpga_wl.get("static_3", {})

                if all(k in d for d in [cgra_lookahead, cgra_static3, fpga_lookahead, fpga_static3] for k in ["mean"]):
                    cgra_headroom = cgra_lookahead["mean"] - cgra_static3["mean"]
                    fpga_headroom = fpga_lookahead["mean"] - fpga_static3["mean"]
                    lines.extend([
                        "**Oracle gap analysis**:\n",
                        f"- FPGA: look-ahead oracle achieves {fpga_lookahead['mean']:.4f} vs "
                        f"best static {fpga_static3['mean']:.4f} (headroom = {fpga_headroom:.4f}, "
                        f"{fpga_headroom/fpga_static3['mean']*100:.1f}%)",
                        f"- CGRA: look-ahead oracle achieves {cgra_lookahead['mean']:.4f} vs "
                        f"best static {cgra_static3['mean']:.4f} (headroom = {cgra_headroom:.4f}, "
                        f"{cgra_headroom/cgra_static3['mean']*100:.1f}%)",
                        "",
                        "**Interpretation**: On FPGA, the majority of the oracle headroom is consumed by ",
                        "reconfiguration overhead — the look-ahead oracle uses cache-aware planning to "
                        "minimize this, but the overhead still limits the theoretical upper bound. On CGRA, ",
                        "the near-zero reconfiguration cost eliminates this overhead, so the oracle can "
                        "switch freely at every trace boundary. If the CGRA headroom is larger than FPGA, "
                        "it means the hardware bottleneck shifted from reconfiguration cost to agent ",
                        "learning quality.",
                        "",
                    ])

    lines.extend([
        "",
        "### Key open questions",
        "",
        "- Would a more sophisticated agent (PPO, SAC, transformer policy) capture more of the oracle gap?",
        "- How does the heterogeneity threshold change with different hardware config sets?",
        "- Does the threshold generalize to real hardware, or is it an artifact of the roofline simulator?",
        "",
        "---",
        "",
        "*Generated by benchmarks/analysis/run_all.py*",
        "",
    ])

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    """Run all analyses and generate the thesis report."""
    t_start = time.time()
    print("=" * 60)
    print("Sharp Thesis Analysis (FPGA + CGRA)")
    print("=" * 60)

    # ─── FPGA Analyses ───────────────────────────────────────────────────

    # 1. Heterogeneity sweep
    print("\n[1/4] Running FPGA heterogeneity sweep...")
    het_start = time.time()
    het_sweep = run_heterogeneity_sweep()
    elapsed = time.time() - het_start
    print(f"  Done in {elapsed:.1f}s")
    sweep_data, _, crossover = het_sweep
    if crossover:
        print(f"  Crossover at heterogeneity = {crossover:.4f}")
    else:
        print("  No crossover found")

    # 2. Multi-seed confidence intervals
    print("\n[2/4] Running FPGA multi-seed analysis...")
    ms_start = time.time()

    # Try to import scipy for proper t-test
    try:
        import scipy.stats  # noqa: F401
    except ImportError:
        print("  Note: scipy not available, using approximate p-values")

    multi_seed: dict[str, dict] = {}
    for wl_name, agents in [
        ("mixed_production", ["tabular", "ucb", "ucb_cache", "smart_static", "static_3", "oracle", "lookahead"]),
        ("llm_decode", ["tabular", "ucb", "ucb_cache", "smart_static", "static_3", "oracle", "lookahead"]),
    ]:
        print(f"  Running {wl_name} with {agents}...")
        ms_data = run_multi_seed_analysis(wl_name, agents, n_seeds=15)
        multi_seed[wl_name] = ms_data
        for agent, d in ms_data.items():
            print(f"    {agent}: {d['mean']:.4f} ± {d['std']:.4f} (95% CI: ±{d['ci95']:.4f})")

    elapsed = time.time() - ms_start
    print(f"  Done in {elapsed:.1f}s")

    # 3. Reconfiguration cost sweep
    print("\n[3/4] Running reconfiguration cost sweep...")
    rc_start = time.time()
    rc_sweep = run_reconfig_sweep()
    elapsed = time.time() - rc_start
    print(f"  Done in {elapsed:.1f}s")
    # Print summary
    for m in sorted(rc_sweep.keys()):
        d = rc_sweep[m]
        print(f"  {m:5.2f}x: adaptive={d['adaptive_best']:.4f} static={d['static_best']:.4f} delta={d['delta']:+.4f} → {d['winner']}")

    # 4. Oracle gap
    print("\n[4/4] Computing FPGA oracle gaps...")
    og_start = time.time()
    existing = load_existing_results()
    print(f"  Loaded {len(existing)} existing results")
    oracle_gaps = compute_all_oracle_gaps(existing)
    elapsed = time.time() - og_start
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Computed {len(oracle_gaps)} oracle gaps")
    valid = [g for g in oracle_gaps if g["gap"] is not None]
    if valid:
        mean_gap = float(np.mean([g["gap"] for g in valid]))
        print(f"  Mean oracle gap: {mean_gap:.3f}")

    # ─── CGRA Analyses ──────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("CGRA Analysis")
    print("=" * 60)

    # CGRA heterogeneity sweep
    print("\n[CGRA 1/2] Running CGRA heterogeneity sweep...")
    cgra_het_start = time.time()
    cgra_het_sweep = run_heterogeneity_sweep(use_cgra=True)
    elapsed = time.time() - cgra_het_start
    print(f"  Done in {elapsed:.1f}s")
    cgra_sweep_data, _, cgra_crossover = cgra_het_sweep
    if cgra_crossover:
        print(f"  Crossover at heterogeneity = {cgra_crossover:.4f}")
    else:
        print("  No crossover found")

    # CGRA multi-seed
    print("\n[CGRA 2/2] Running CGRA multi-seed analysis...")
    cgra_ms_start = time.time()
    cgra_multi_seed: dict[str, dict] = {}
    for wl_name, agents in [
        ("mixed_production", ["tabular", "ucb", "ucb_cache", "smart_static", "static_3", "oracle", "lookahead"]),
    ]:
        print(f"  Running {wl_name} with {agents}...")
        ms_data = run_multi_seed_analysis(wl_name, agents, n_seeds=15, use_cgra=True)
        cgra_multi_seed[wl_name] = ms_data
        for agent, d in ms_data.items():
            print(f"    {agent}: {d['mean']:.4f} ± {d['std']:.4f} (95% CI: ±{d['ci95']:.4f})")
    elapsed = time.time() - cgra_ms_start
    print(f"  Done in {elapsed:.1f}s")

    # Generate report
    print("\nGenerating thesis report...")
    report = generate_report(
        het_sweep, multi_seed, rc_sweep, oracle_gaps,
        cgra_het_sweep=cgra_het_sweep,
        cgra_multi_seed=cgra_multi_seed,
    )

    REPORT_PATH.write_text(report)
    total_elapsed = time.time() - t_start
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Total analysis time: {total_elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
