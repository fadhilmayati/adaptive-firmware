"""CGRA (Coarse-Grained Reconfigurable Array) accelerator configurations.

CGRAs are a fundamentally different architecture from FPGA:
- Reconfiguration is cycle-level (~1ns) instead of millisecond-level (~3-8ms)
- PE arrays with ALUs/MACs instead of LUT-based fabric
- Multi-context configuration memory stores several mappings on-chip
- Higher compute density and energy efficiency than FPGA for the same ops

These configs mirror the FPGA CONFIG_PRESETS in naming convention (WIDE → COMPUTE,
STREAM → BANDWIDTH, etc.) but with different absolute numbers reflecting CGRA
characteristics: higher throughput, lower energy, near-zero reconfiguration cost.
"""

from __future__ import annotations

from .configs import AcceleratorConfig


# All CGRA configs have near-zero reconfiguration time (single cycle at 1GHz)
CGRA_RECONFIG_MS = 0.001

# Cache capacity: CGRAs typically have 4-16 configuration contexts on-chip.
# Set to 4 so all configs are always loaded — no cache misses.
CGRA_CACHE_CAPACITY = 4

CGRA_PRESETS: list[AcceleratorConfig] = [
    AcceleratorConfig(
        config_id=0,
        name="CGRA_WIDE",
        compute_throughput=1200.0,   # 16x16 PE array, 32-bit → very high compute
        memory_bandwidth=40.0,       # Limited by chip I/O
        energy_per_op=8.0,           # Wide data paths cost more energy
        reconfig_time_ms=CGRA_RECONFIG_MS,
        optimal_for="compute_bound",
    ),
    AcceleratorConfig(
        config_id=1,
        name="CGRA_STREAM",
        compute_throughput=500.0,    # 8x8 PE array
        memory_bandwidth=160.0,      # Dedicated streaming ports for high bandwidth
        energy_per_op=6.0,           # Smaller array, more efficient
        reconfig_time_ms=CGRA_RECONFIG_MS,
        optimal_for="memory_bound",
    ),
    AcceleratorConfig(
        config_id=2,
        name="CGRA_BAL",
        compute_throughput=800.0,    # 8x8 PE array, 16-bit
        memory_bandwidth=80.0,       # Balanced memory access
        energy_per_op=7.0,           # Efficient 16-bit operations
        reconfig_time_ms=CGRA_RECONFIG_MS,
        optimal_for="balanced",
    ),
    AcceleratorConfig(
        config_id=3,
        name="CGRA_LP",
        compute_throughput=300.0,    # 4x4 PE array, 8-bit
        memory_bandwidth=30.0,       # Minimal bandwidth
        energy_per_op=3.0,           # Very low power
        reconfig_time_ms=CGRA_RECONFIG_MS,
        optimal_for="low_power",
    ),
]
