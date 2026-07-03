#!/usr/bin/env python3
"""Demo: Adaptive firmware layer PoC.

Demonstrates the full closed loop:
1. Generate mixed workloads (compute-bound, memory-bound, balanced)
2. Run the adaptive agent — it learns to reconfigure for each workload phase
3. Run static baselines (each fixed config) for comparison
4. Show that the adaptive agent outperforms any single static config

This is the proof-of-concept that the adaptive firmware layer works.
"""

import sys
import os

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adaptive_firmware.hardware.configs import CONFIG_PRESETS
from adaptive_firmware.runtime.middleware import AdaptiveMiddleware
from adaptive_firmware.workloads.models import create_synthetic_traces


def run_demo():
    print("=" * 72)
    print("  Adaptive Firmware Layer — PoC Demo")
    print("  RL-driven runtime reconfiguration of simulated silicon")
    print("=" * 72)

    # 1. Generate mixed workload traces
    # 3 phases: compute-bound → memory-bound → balanced, repeated 3 times
    print("\n[1] Generating synthetic workload traces...")
    traces = create_synthetic_traces(
        n_compute=30,
        n_memory=30,
        n_balanced=30,
    )
    print(f"    Generated {len(traces)} traces across 3 workload classes:")
    compute_count = sum(1 for t in traces if t.workload_class == "compute_bound")
    memory_count = sum(1 for t in traces if t.workload_class == "memory_bound")
    balanced_count = sum(1 for t in traces if t.workload_class == "balanced")
    print(f"    - compute_bound: {compute_count}")
    print(f"    - memory_bound: {memory_count}")
    print(f"    - balanced: {balanced_count}")

    # Shuffle to create a realistic mixed workload
    import random
    random.seed(42)
    random.shuffle(traces)
    print(f"    (shuffled to simulate mixed workload)")

    # 2. Run adaptive agent
    print("\n[2] Running adaptive agent (online learning)...")
    print("    (agent starts with no knowledge, learns from reward feedback)")
    mw = AdaptiveMiddleware(
        configs=CONFIG_PRESETS,
        cache_capacity=2,
        learning_rate=0.15,
        epsilon_start=0.4,
        verbose=False,
    )
    adaptive_report = mw.run_episode(traces)

    print(f"\n  Adaptive Agent Results:")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ {adaptive_report.summary_str()}")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Show learning curve (avg reward in first third vs last third)
    n = len(adaptive_report.logs)
    first_third = adaptive_report.logs[: n // 3]
    last_third = adaptive_report.logs[2 * n // 3 :]
    early_reward = sum(l.reward for l in first_third) / max(len(first_third), 1)
    late_reward = sum(l.reward for l in last_third) / max(len(last_third), 1)
    print(f"\n  Learning curve:")
    print(f"    Early avg reward (first 1/3):  {early_reward:.4f}")
    print(f"    Late avg reward  (last 1/3):   {late_reward:.4f}")
    print(f"    Improvement:                   {((late_reward - early_reward) / max(early_reward, 0.001) * 100):.1f}%")

    # 3. Run static baselines
    print("\n[3] Running static baselines (fixed config, no adaptation)...")
    static_results = {}
    for config in CONFIG_PRESETS:
        mw_static = AdaptiveMiddleware(configs=CONFIG_PRESETS, cache_capacity=2)
        report = mw_static.run_static_baseline(traces, config.config_id)
        static_results[config.name] = report
        print(
            f"    {config.name:16s}: "
            f"time={report.total_time_ms:10.2f}ms  "
            f"energy={report.total_energy_mj:10.2f}mJ  "
            f"reward={report.avg_reward:.4f}"
        )

    # 4. Compare
    print("\n[4] Comparison: Adaptive vs Static")
    print("  " + "-" * 66)

    best_static_name = max(static_results, key=lambda k: static_results[k].avg_reward)
    best_static = static_results[best_static_name]

    print(f"  {'Metric':<20s} {'Adaptive':>12s} {'Best Static':>12s} {'Delta':>10s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    print(
        f"  {'Avg reward':<20s} "
        f"{adaptive_report.avg_reward:>12.4f} "
        f"{best_static.avg_reward:>12.4f} "
        f"{(adaptive_report.avg_reward - best_static.avg_reward):>+10.4f}"
    )
    print(
        f"  {'Total time (ms)':<20s} "
        f"{adaptive_report.total_time_ms:>12.2f} "
        f"{best_static.total_time_ms:>12.2f} "
        f"{(adaptive_report.total_time_ms - best_static.total_time_ms):>+10.2f}"
    )
    print(
        f"  {'Total energy (mJ)':<20s} "
        f"{adaptive_report.total_energy_mj:>12.2f} "
        f"{best_static.total_energy_mj:>12.2f} "
        f"{(adaptive_report.total_energy_mj - best_static.total_energy_mj):>+10.2f}"
    )
    print(
        f"  {'Cache hit rate':<20s} "
        f"{adaptive_report.cache_hit_rate*100:>11.1f}% "
        f"{best_static.cache_hit_rate*100:>11.1f}% "
        f"{'':>10s}"
    )

    # 5. Final policy
    print(f"\n[5] Agent's learned policy:")
    print(f"  The agent learned which config to use for each workload class:")
    config_names = {c.config_id: c.name for c in CONFIG_PRESETS}
    for state, action in adaptive_report.final_policy.items():
        print(f"    {state:16s} -> {config_names.get(action, '???')}")

    print(f"\n  Config usage distribution:")
    for cid, count in sorted(adaptive_report.config_usage.items()):
        bar = "#" * (count // 3)
        print(f"    {config_names.get(cid, '???'):16s}: {count:4d} {bar}")

    # 6. Verdict
    print(f"\n{'=' * 72}")
    if adaptive_report.avg_reward > best_static.avg_reward:
        print(f"  ✓ VERDICT: Adaptive agent OUTPERFORMS best static config")
        print(f"    Reward improvement: +{(adaptive_report.avg_reward - best_static.avg_reward):.4f}")
        print(f"    Learning improvement: {((late_reward - early_reward) / max(early_reward, 0.001) * 100):.1f}%")
        print(f"\n  The adaptive firmware layer successfully:")
        print(f"    1. Observed mixed workloads (compute/memory/balanced)")
        print(f"    2. Learned online which config to use for each workload type")
        print(f"    3. Outperformed every static configuration")
        print(f"    4. Detected {adaptive_report.drift_count} workload drift events")
        print(f"\n  PoC SUCCESSFUL — the concept is validated.")
    else:
        print(f"  ✗ VERDICT: Adaptive agent did NOT outperform best static config")
        print(f"    This may happen with short episodes or too much exploration.")
        print(f"    Try increasing the number of traces or reducing epsilon_start.")
    print(f"{'=' * 72}")


def run_multi_tenant_demo():
    """Run a multi-tenant demo showing the agent handling concurrent workloads."""
    print("\n\n" + "=" * 72)
    print("  Multi-Tenant Demo")
    print("=" * 72)

    # Create two tenants with different workload profiles
    tenant_cnn = create_synthetic_traces(n_compute=40, n_memory=0, n_balanced=10)
    tenant_mem = create_synthetic_traces(n_compute=0, n_memory=40, n_balanced=10)

    import random
    random.seed(123)
    random.shuffle(tenant_cnn)
    random.shuffle(tenant_mem)

    print(f"\n  Tenant 'cnn_service': {len(tenant_cnn)} traces (mostly compute-bound)")
    print(f"  Tenant 'embed_service': {len(tenant_mem)} traces (mostly memory-bound)")

    mw = AdaptiveMiddleware(
        configs=CONFIG_PRESETS,
        cache_capacity=2,
        learning_rate=0.15,
        epsilon_start=0.4,
    )

    report = mw.run_multi_tenant({
        "cnn_service": tenant_cnn,
        "embed_service": tenant_mem,
    }, interleave=True)

    print(f"\n  Multi-Tenant Results:")
    print(f"  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ {report.summary_str()}")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Show per-tenant breakdown
    cnn_logs = [l for l in report.logs if l.tenant_id == "cnn_service"]
    mem_logs = [l for l in report.logs if l.tenant_id == "embed_service"]

    print(f"\n  Per-tenant breakdown:")
    for tid, logs in [("cnn_service", cnn_logs), ("embed_service", mem_logs)]:
        if not logs:
            continue
        avg_r = sum(l.reward for l in logs) / len(logs)
        config_dist = {}
        for l in logs:
            config_dist[l.config_name] = config_dist.get(l.config_name, 0) + 1
        print(f"    {tid}: avg_reward={avg_r:.4f}, configs={config_dist}")

    print(f"\n  Final policy: {report.final_policy}")
    print(f"  The agent learned to serve different configs to different tenants.")


if __name__ == "__main__":
    run_demo()
    run_multi_tenant_demo()
