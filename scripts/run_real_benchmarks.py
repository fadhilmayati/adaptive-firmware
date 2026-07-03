#!/usr/bin/env python3
"""Real workload benchmarks for the adaptive firmware layer.

Runs the adaptive agent against three real-world workload patterns:
1. LLM token streaming: compute-bound prefill + memory-bound decode
2. YOLO detection: compute-bound backbone + memory-bound head
3. Whisper encoder: compute-bound conv frontend + memory-bound attention

Compares adaptive vs best static config, and characterizes where
adaptation helps most vs where it doesn't.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adaptive_firmware.workloads.real_models import (
    run_all_benchmarks,
    run_mixed_production_benchmark,
    run_workload_benchmark,
    generate_llm_traces,
    generate_yolo_traces,
    generate_whisper_traces,
)


def print_benchmark_result(r) -> None:
    """Pretty-print a single benchmark result."""
    print(f"\n{'─' * 70}")
    print(f"  {r.name}")
    print(f"{'─' * 70}")
    print(f"  Traces: {r.n_traces}")
    print(f"  Workload class distribution:")
    for cls, count in sorted(r.workload_class_distribution.items()):
        pct = 100 * count / r.n_traces
        bar = "#" * int(pct / 2)
        print(f"    {cls:16s}: {count:4d} ({pct:5.1f}%) {bar}")

    print(f"\n  {'Metric':<22s} {'Adaptive':>12s} {'Best Static':>12s} {'Delta':>12s}")
    print(f"  {'─' * 22} {'─' * 12} {'─' * 12} {'─' * 12}")
    print(
        f"  {'Avg reward':<22s} "
        f"{r.adaptive_reward:>12.4f} "
        f"{r.best_static_reward:>12.4f} "
        f"{r.reward_improvement:>+12.4f}"
    )
    print(
        f"  {'Total time (ms)':<22s} "
        f"{r.adaptive_time_ms:>12.2f} "
        f"{r.best_static_time_ms:>12.2f} "
        f"{(r.adaptive_time_ms - r.best_static_time_ms):>+12.2f}"
    )
    print(
        f"  {'Total energy (mJ)':<22s} "
        f"{r.adaptive_energy_mj:>12.2f} "
        f"{r.best_static_energy_mj:>12.2f} "
        f"{(r.adaptive_energy_mj - r.best_static_energy_mj):>+12.2f}"
    )
    print(f"  {'Best static config':<22s} {'':>12s} {r.best_static_name:>12s} {'':>12s}")
    print(f"  {'Cache hit rate':<22s} {r.adaptive_cache_hit_rate * 100:>11.1f}% {'':>12s} {'':>12s}")

    if r.reward_improvement > 0:
        pct = r.reward_improvement / max(r.best_static_reward, 0.001) * 100
        print(f"\n  ✓ Adaptive wins by {r.reward_improvement:+.4f} ({pct:+.1f}%)")
    else:
        print(f"\n  ✗ Adaptive loses by {r.reward_improvement:+.4f}")


def main() -> None:
    print("=" * 70)
    print("  Real Workload Benchmarks — Adaptive Firmware Layer")
    print("  Three workload patterns from production AI:")
    print("    1. LLM token streaming (prefill + decode)")
    print("    2. YOLO object detection (backbone + head)")
    print("    3. Whisper audio encoder (conv + transformer)")
    print("=" * 70)

    results = run_all_benchmarks()

    mixed = run_mixed_production_benchmark(n_inferences=200, seed=42)
    results.append(mixed)

    for r in results:
        print_benchmark_result(r)

    # Summary
    print(f"\n{'═' * 70}")
    print("  Summary: Where does adaptation help?")
    print(f"{'═' * 70}")
    print(f"  {'Workload':<25s} {'Adaptive':>10s} {'Static':>10s} {'Win?':>8s}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 8}")
    adaptive_wins = 0
    for r in results:
        win = "✓ YES" if r.reward_improvement > 0 else "✗ no"
        if r.reward_improvement > 0:
            adaptive_wins += 1
        print(
            f"  {r.name:<25s} "
            f"{r.adaptive_reward:>10.4f} "
            f"{r.best_static_reward:>10.4f} "
            f"{win:>8s}"
        )

    print(f"\n  Adaptive wins: {adaptive_wins}/{len(results)}")

    # Characterization insight
    print(f"\n  Insight:")
    if adaptive_wins == len(results):
        print("    Adaptation helps across all workload types tested.")
    elif adaptive_wins > 0:
        print("    Adaptation helps most on workloads with mixed compute/memory")
        print("    patterns AND sufficient length to amortize exploration cost.")
        print("    Short, homogeneous workloads (YOLO, Whisper) see less benefit.")
    else:
        print("    Static configs are competitive — the simulator may not be")
        print("    modeling the dynamic cost of reconfiguration realistically.")


if __name__ == "__main__":
    main()
