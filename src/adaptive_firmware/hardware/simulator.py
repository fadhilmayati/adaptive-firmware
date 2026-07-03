"""Hardware simulator for reconfigurable accelerator.

Models a single reconfigurable region that can load different accelerator
configurations (partial bitstreams). Given a workload (FLOPs, memory bytes),
it computes execution time, energy consumed, and whether the workload is
compute-bound or memory-bound — which the agent uses to decide reconfiguration.

The simulator is intentionally simple but captures the key tradeoffs:
- Compute-bound workloads benefit from HIGH_COMPUTE config
- Memory-bound workloads benefit from HIGH_BANDWIDTH config
- Balanced workloads work well with BALANCED config
- Low-power scenarios benefit from LOW_POWER config
- Reconfiguration has a cost (time + energy) that must be amortized
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .configs import AcceleratorConfig
from .bitstream_cache import BitstreamCache


@dataclass
class ExecutionResult:
    """Result of executing a workload on a specific config."""

    exec_time_ms: float
    energy_mj: float
    reconfig_time_ms: float
    total_time_ms: float  # exec_time + reconfig_time
    throughput_gops: float
    config_id: int
    cache_hit: bool


class HardwareSimulator:
    """Simulates a reconfigurable accelerator with a bitstream cache.

    The simulator takes a workload (defined by FLOPs and memory bytes)
    and an accelerator config, then computes:
    - How long the compute takes (limited by compute throughput)
    - How long the memory access takes (limited by memory bandwidth)
    - The actual execution time (max of the two, roofline model)
    - Energy consumed
    - Reconfiguration overhead (from bitstream cache)
    """

    def __init__(
        self,
        configs: list[AcceleratorConfig],
        cache_capacity: int = 2,
        idle_power_mw: float = 200.0,
    ) -> None:
        """Initialize the simulator.

        Args:
            configs: Available accelerator configurations.
            cache_capacity: Number of PRRs (simultaneous loaded bitstreams).
            idle_power_mw: Power draw when idle (for reconfig energy model).
        """
        self.configs = {c.config_id: c for c in configs}
        self.cache = BitstreamCache(capacity=cache_capacity)
        self.idle_power_mw = idle_power_mw
        self.current_config_id: int | None = None
        self.total_exec_time_ms: float = 0.0
        self.total_energy_mj: float = 0.0
        self.total_reconfig_time_ms: float = 0.0

    def execute(
        self,
        flops: float,
        memory_bytes: float,
        config_id: int,
    ) -> ExecutionResult:
        """Execute a workload on a specific config.

        Uses a roofline model: execution time is the max of compute time
        and memory time, reflecting whether the workload is compute-bound
        or memory-bound on this particular config.

        Args:
            flops: Number of floating-point operations (e.g., 1e9 = 1 GFLOP).
            memory_bytes: Bytes of memory accessed (e.g., 1e9 = 1 GB).
            config_id: Which accelerator config to use.

        Returns:
            ExecutionResult with timing and energy breakdown.
        """
        if config_id not in self.configs:
            raise ValueError(f"Unknown config_id: {config_id}")

        config = self.configs[config_id]

        # Roofline model
        # Convert GOPS to ops/s, GB/s to bytes/s
        compute_time_s = flops / (config.compute_throughput * 1e9)
        memory_time_s = memory_bytes / (config.memory_bandwidth * 1e9)

        exec_time_s = max(compute_time_s, memory_time_s)
        exec_time_ms = exec_time_s * 1000.0

        # Energy: compute energy + memory access energy + idle during reconfig
        compute_energy_mj = (flops * config.energy_per_op) * 1e-12 * 1000.0  # pJ->mJ
        # Memory energy: ~4 pJ/byte (simplified DRAM access cost)
        memory_energy_mj = (memory_bytes * 4.0) * 1e-12 * 1000.0

        # Reconfiguration overhead
        reconfig_time_ms = self.cache.request(config)
        reconfig_energy_mj = (reconfig_time_ms / 1000.0) * self.idle_power_mw

        total_time_ms = exec_time_ms + reconfig_time_ms
        total_energy_mj = compute_energy_mj + memory_energy_mj + reconfig_energy_mj

        # Throughput achieved
        throughput_gops = (flops / 1e9) / (exec_time_s if exec_time_s > 0 else 1e-9)

        self.current_config_id = config_id
        self.total_exec_time_ms += exec_time_ms
        self.total_energy_mj += total_energy_mj
        self.total_reconfig_time_ms += reconfig_time_ms

        return ExecutionResult(
            exec_time_ms=exec_time_ms,
            energy_mj=total_energy_mj,
            reconfig_time_ms=reconfig_time_ms,
            total_time_ms=total_time_ms,
            throughput_gops=throughput_gops,
            config_id=config_id,
            cache_hit=(reconfig_time_ms == 0.0),
        )

    @staticmethod
    def classify_workload(flops: float, memory_bytes: float) -> str:
        """Classify a workload as compute-bound, memory-bound, or balanced.

        Uses the arithmetic intensity (FLOPs per byte) as the discriminator:
        - AI < 5: memory-bound (data movement dominates)
        - AI > 20: compute-bound (compute dominates)
        - 5 <= AI <= 20: balanced
        """
        if memory_bytes == 0:
            return "compute_bound"
        ai = flops / memory_bytes  # arithmetic intensity
        if ai < 5.0:
            return "memory_bound"
        elif ai > 20.0:
            return "compute_bound"
        else:
            return "balanced"

    def reset(self) -> None:
        """Reset simulator state (for new episodes)."""
        self.cache.reset()
        self.current_config_id = None
        self.total_exec_time_ms = 0.0
        self.total_energy_mj = 0.0
        self.total_reconfig_time_ms = 0.0
