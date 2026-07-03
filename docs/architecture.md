# Architecture

This document describes the internal architecture of the adaptive firmware
layer. If you're looking for the high-level story, see [STATUS.md](../STATUS.md).
If you want to verify the underlying research claims, see
[concepts/workload-adaptive-silicon-verification.md](../concepts/workload-adaptive-silicon-verification.md).

## Overview

The adaptive firmware layer sits between an AI runtime (PyTorch) and
reconfigurable silicon (simulated FPGA/CGRA). The core closed loop:

```
PyTorch model
    ↓ (forward hooks)
Observation layer
    ↓ (telemetry vector)
RL agent
    ↓ (config_id)
Bitstream cache + Hardware simulator
    ↓ (ExecutionResult)
Reward computation
    ↓ (reward)
RL agent (update)
```

## Module breakdown

### `src/adaptive_firmware/hardware/` — Simulated silicon

The hardware layer models a reconfigurable accelerator with a bitstream cache.

**`configs.py`** — Defines `AcceleratorConfig` (compute throughput, memory
bandwidth, energy per op, reconfiguration time) and four presets:

- `HIGH_COMPUTE`: 800 GOPS, 30 GB/s, 12 pJ/op, 8ms reconfig
- `HIGH_BANDWIDTH`: 300 GOPS, 120 GB/s, 8 pJ/op, 6ms reconfig
- `BALANCED`: 500 GOPS, 60 GB/s, 10 pJ/op, 5ms reconfig
- `LOW_POWER`: 200 GOPS, 40 GB/s, 4 pJ/op, 3ms reconfig

These represent the Pareto frontier of compute/memory/energy tradeoffs that
real reconfigurable chips expose.

**`bitstream_cache.py`** — LRU cache that holds currently loaded configs. A
cache hit means no reconfiguration cost; a miss incurs the config's load
time. Capacity defaults to 2 (two PRRs).

**`simulator.py`** — Uses the roofline model. Execution time is the max of
compute time (`flops / compute_throughput`) and memory time
(`memory_bytes / memory_bandwidth`). Energy is the sum of compute energy
(flops × energy_per_op), memory energy (bytes × 4 pJ/byte), and reconfig
energy (reconfig_time × idle_power).

### `src/adaptive_firmware/observation/` — AI runtime → telemetry

**`pytorch_hooks.py`** — Attaches forward hooks to PyTorch modules
(Conv2d, Linear, MultiheadAttention). On each forward pass, estimates FLOPs
and memory accessed for that op. FLOP estimation uses standard formulas
(2 × out_ch × out_h × out_w × in_ch × k_h × k_w for Conv2d).

**`telemetry.py`** — Defines `WorkloadTrace` (the raw op-level data) and
`TelemetryVector` (the 8-dimensional feature vector the agent sees). The
workload class is derived from arithmetic intensity: `flops / memory_bytes`
→ compute_bound (>20), memory_bound (<5), or balanced (5-20).

### `src/adaptive_firmware/agent/` — The learning brain

Four agent implementations, each with different tradeoffs:

**`rl_agent.py`** — `ReconfigAgent`: Tabular Q-learning with informed
initialization. Fast to converge, but doesn't scale to continuous state
spaces. The baseline.

**`neural_agent.py`** — `NeuralReconfigAgent`: Small NumPy MLP (2 hidden
layers, 64 units, ~10K params) with experience replay and target network.
More powerful in theory, harder to train in practice.

**`lookahead_agent.py`** — `LookaheadAgent`: Looks at the next 5 workload
classes, classifies the pattern (single/stable/changing/cycling), and
prefetches configs to overlap reconfiguration with execution. Reduces
reconfig overhead by 21%.

**`drift_detector.py`** — `DriftDetector`: ADWIN-inspired window-based
detector. When the workload distribution changes, boosts exploration so the
agent can re-learn.

### `src/adaptive_firmware/runtime/` — The middleware

**`middleware.py`** — The `AdaptiveMiddleware` class orchestrates the full
closed loop. Supports single-tenant (`run_episode`) and multi-tenant
(`run_multi_tenant`) modes. The latter interleaves workload traces from
multiple tenants and the agent arbitrates config selection across all.

The `StepLog` dataclass records each step's decision and result. The
`EpisodeReport` aggregates results for a full run.

### `src/adaptive_firmware/workloads/` — Test scenarios

**`models.py`** — `SimpleCNN`, `SimpleMLP`, `MemoryHeavyModel`: PyTorch
models with known workload characteristics, plus `create_workload_sequence`
and `create_synthetic_traces` for fast testing.

**`real_models.py`** — Trace generators for real production patterns:
- `generate_llm_traces`: LLM token streaming (prefill + decode, growing KV cache)
- `generate_yolo_traces`: YOLO detection (compute-bound backbone + memory-bound head)
- `generate_whisper_traces`: Whisper encoder (conv frontend + transformer)
- `run_mixed_production_benchmark`: simulates a production AI serving system

## Data flow

A single step of the closed loop:

1. **Observe**: `PyTorchObserver` captures a `WorkloadTrace` (op type, FLOPs,
   memory, tensor shapes). The middleware builds a `TelemetryVector` from
   this plus the current hardware state (loaded config, cache contents).

2. **Decide**: `ReconfigAgent.select_action(telemetry)` returns a config_id
   via epsilon-greedy over Q-values. The Q-table is seeded with workload-class
   priors so the agent starts with reasonable beliefs.

3. **Execute**: `HardwareSimulator.execute(flops, memory_bytes, config_id)`
   runs the roofline model, computes execution time and energy, and records
   the result in `ExecutionResult`. The bitstream cache handles the
   reconfig cost transparently.

4. **Measure**: `ReconfigAgent.compute_reward(result)` returns a reward in
   [0, 1] balancing throughput, energy efficiency, cache hit bonus, and
   reconfig penalty.

5. **Learn**: `ReconfigAgent.update(telemetry, action, reward)` performs the
   Q-learning update and decays epsilon. The drift detector monitors the
   reward stream and boosts exploration if the workload changes.

## Key design decisions

**Why a roofline simulator, not cycle-accurate?**
We wanted to validate the concept before investing in cycle-accurate modeling.
The roofline captures the essential compute/memory tradeoff that drives
reconfiguration decisions. Cycle-accurate modeling (e.g., via sarchlab/zeonica)
is a roadmap item.

**Why a contextual bandit, not full RL?**
Each reconfiguration decision is independent with immediate reward. There's
no sequential dependency between decisions. Full RL (PPO, SAC) would add
complexity without benefit.

**Why informed initialization?**
The Q-table is seeded with workload-class priors (e.g.,
"compute_bound → HIGH_COMPUTE"). This eliminates the cold-start cost of
discovering obvious mappings and lets the agent focus exploration on nuanced
cases.

**Why a small MLP, not a transformer?**
The state space is 8 features, the action space is 4 configs. A small MLP
runs in microseconds on CPU, well within the 1-15ms loop budget. The
neural network's value is in continuous state generalization and transfer
learning, not in modeling complex sequences.

**Why look-ahead scheduling?**
Reconfiguration is expensive (3-8ms per bitstream). If the agent can see
that a different config will be needed in 3 ops, it can start loading it
now and hide the cost behind the current op's execution. This is where
the loop time budget starts to matter.

## Testing

36 tests across 4 files:

- `tests/test_hardware_sim.py` (9 tests) — bitstream cache, roofline model, workload classification
- `tests/test_agent.py` (9 tests) — Q-learning, drift detection, reward computation
- `tests/test_neural_agent.py` (12 tests) — MLP forward/backward, replay buffer, target network
- `tests/test_e2e.py` (6 tests) — full middleware loop, multi-tenant, static baseline comparison

## Future architecture

**Real hardware port (Kria KV260, $249)**: Replace the `HardwareSimulator`
with a Linux FPGA Manager backend that calls `fpga_mgr_write` on the actual
device. The rest of the architecture stays the same.

**CGRA substrate**: Add a `CGRASimulator` alongside `HardwareSimulator`.
The middleware already handles config selection generically; adding a
substrate is a matter of implementing the same interface (execute,
reconfig_time, etc.) for CGRA semantics.

**PyTorch operator-level dispatch**: Move from forward hooks to intercepting
at the `torch.ops` level. This lets the agent see the operator graph
ahead of execution, enabling better look-ahead scheduling.

**Standardized benchmark suite**: A `benchmarks/` directory with
reproducible workloads, a runner, and a results format. Anyone working in
this space can run the same benchmarks and compare.
