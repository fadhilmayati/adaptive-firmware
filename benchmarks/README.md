# Adaptive Firmware Layer — Benchmark Suite

A standardized benchmark suite for adaptive reconfigurable hardware. Use it to:
- Run reproducible workloads on any adaptive agent
- Compare your agent against the baselines
- Submit results in a standard format

The field of adaptive reconfigurable hardware doesn't have a standard benchmark. This is the first.

## Quick start

```bash
# Run the full suite (5 workloads × 6 agents = 30 benchmarks)
python -m benchmarks.runner

# Run a single workload with a specific agent
python -c "from benchmarks.runner import run_workload; print(run_workload('llm_decode', agent='tabular').to_json())"

# Generate the leaderboard
python -c "from benchmarks.aggregator import generate_leaderboard; print(generate_leaderboard())"
```

## Available workloads

| Workload | Version | Description | Traces |
|----------|---------|-------------|--------|
| `llm_decode` | 1.0.0 | LLM token streaming decode phase: growing KV cache, memory-bound | 160 |
| `llm_prefill` | 1.0.0 | LLM token streaming prefill phase: parallel processing, compute-bound | 32 |
| `cv_detection` | 1.0.0 | YOLO-style detection: compute-bound backbone + memory-bound head | 6 |
| `audio_encoder` | 1.0.0 | Whisper-style encoder: compute-bound conv + memory-bound transformer | 6 |
| `mixed_production` | 1.0.0 | Production AI serving: 40% LLM decode, 25% CV, 20% audio, 15% prefill | ~600 |

All workloads are seeded and reproducible. Same seed → same traces.

## Available agents

| Agent | Description |
|-------|-------------|
| `tabular` | Tabular Q-learning with informed initialization (our baseline) |
| `neural` | Small NumPy MLP with experience replay and target network |
| `oracle` | Always picks the best config for the workload class (upper bound) |
| `static_0` | HIGH_COMPUTE always |
| `static_1` | HIGH_BANDWIDTH always |
| `static_2` | BALANCED always |
| `static_3` | LOW_POWER always |
| `random` | Random config each step (sanity check) |

## Adding a new workload

1. Create `benchmarks/workloads/my_workload.py`
2. Define a `generate_traces(seed)` function
3. Register a `WorkloadSpec` at module level
4. Import the module in `benchmarks/workloads/__init__.py`

```python
from .base import WorkloadSpec
from .registry import register_workload

def generate_traces(seed: int = 42) -> list[WorkloadTrace]:
    # Your trace generation logic here
    ...

register_workload(WorkloadSpec(
    name="my_workload",
    version="1.0.0",
    description="What this workload represents",
    tags=["tag1", "tag2"],
    workload_class="single-tenant",  # or "multi-tenant"
    trace_generator=generate_traces,
    expected_n_traces=100,  # for validation
    seed=42,
))
```

## Adding a new agent

1. Create a new `run()` method or extend the runner
2. Add the agent name to `BenchmarkRunner._build_middleware`
3. Document it in this README

## Results format

Results are written as JSON to `benchmarks/results/`. Each file contains:

```json
{
  "suite_version": "0.1.0",
  "timestamp": "2026-07-03T...",
  "workload_name": "llm_decode",
  "workload_version": "1.0.0",
  "agent_name": "tabular",
  "agent_config": {},
  "n_traces": 160,
  "avg_reward": 0.42,
  "total_time_ms": 1234.5,
  "total_energy_mj": 567.8,
  "total_reconfig_time_ms": 12.3,
  "cache_hit_rate": 0.85,
  "config_usage": {"0": 10, "1": 20, ...},
  "final_policy": {"compute_bound": 0, "memory_bound": 1, ...},
  "run_time_seconds": 2.3
}
```

This format is versioned. Adding fields is fine. Renaming or removing fields requires a suite version bump.

## Reproducibility

Every workload is seeded. Every agent has a fixed seed. Running the same suite twice produces identical JSON files (modulo wall-clock time fields).

```bash
# Run twice, diff
python -m benchmarks.runner
python -m benchmarks.runner
diff -r benchmarks/results/ /tmp/baseline_results/
```

The only non-deterministic field is `run_time_seconds` (wall-clock).

## Submitting your results

1. Fork the repo
2. Add your results JSON files to `benchmarks/results/`
3. Open a PR with:
   - Your agent's code
   - A description of what makes it different
   - Your results JSON files
4. We'll add you to the leaderboard

## What this is NOT

- Not a microbenchmark — we measure end-to-end reward, not individual op latency
- Not a hardware-specific benchmark — we run in simulation, on any CPU
- Not a single-number ranking — different workloads favor different approaches

## License

MIT, same as the rest of this project.
