"""Accelerator configurations for the simulated reconfigurable hardware.

Each config represents a partial bitstream that can be loaded into a
reconfigurable region (PRR). Different configs trade off compute throughput,
memory bandwidth, and energy efficiency — the agent's job is to pick the
right one for the current workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AcceleratorConfig:
    """A single reconfigurable accelerator configuration (partial bitstream).

    Attributes:
        config_id: Unique identifier (used as RL action).
        name: Human-readable name.
        compute_throughput: GOPS (giga-ops per second) when running.
        memory_bandwidth: GB/s available to this config.
        energy_per_op: picojoules per operation.
        reconfig_time_ms: Time to load this bitstream into a PRR (milliseconds).
        optimal_for: Workload profile this config is tuned for ("compute_bound",
                     "memory_bound", "balanced", "low_power").
    """

    config_id: int
    name: str
    compute_throughput: float  # GOPS
    memory_bandwidth: float    # GB/s
    energy_per_op: float       # pJ/op
    reconfig_time_ms: float    # ms to load bitstream
    optimal_for: str           # workload profile tag


CONFIG_PRESETS: list[AcceleratorConfig] = [
    AcceleratorConfig(
        config_id=0,
        name="HIGH_COMPUTE",
        compute_throughput=800.0,
        memory_bandwidth=30.0,
        energy_per_op=12.0,
        reconfig_time_ms=8.0,
        optimal_for="compute_bound",
    ),
    AcceleratorConfig(
        config_id=1,
        name="HIGH_BANDWIDTH",
        compute_throughput=300.0,
        memory_bandwidth=120.0,
        energy_per_op=8.0,
        reconfig_time_ms=6.0,
        optimal_for="memory_bound",
    ),
    AcceleratorConfig(
        config_id=2,
        name="BALANCED",
        compute_throughput=500.0,
        memory_bandwidth=60.0,
        energy_per_op=10.0,
        reconfig_time_ms=5.0,
        optimal_for="balanced",
    ),
    AcceleratorConfig(
        config_id=3,
        name="LOW_POWER",
        compute_throughput=200.0,
        memory_bandwidth=40.0,
        energy_per_op=4.0,
        reconfig_time_ms=3.0,
        optimal_for="low_power",
    ),
]
