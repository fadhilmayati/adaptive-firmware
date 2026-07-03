"""Telemetry data structures for the adaptive firmware layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkloadTrace:
    """A single observation of a workload passing through the system.

    Captured by PyTorch forward hooks. Contains everything the RL agent
    needs to make a reconfiguration decision.
    """

    op_type: str              # e.g., "Conv2d", "Linear", "MatMul"
    flops: float              # estimated FLOPs for this op
    memory_bytes: float       # estimated memory accessed (bytes)
    batch_size: int           # batch size
    tensor_shapes: list[tuple[int, ...]]  # input tensor shapes
    arithmetic_intensity: float  # flops / memory_bytes
    workload_class: str       # "compute_bound" | "memory_bound" | "balanced"


@dataclass
class TelemetryVector:
    """The state vector the RL agent sees.

    Combines workload trace info with simulated hardware telemetry.
    This is what gets fed into the agent's policy for each decision.
    """

    # Workload features
    op_type: str
    flops: float
    memory_bytes: float
    arithmetic_intensity: float
    workload_class: str

    # Hardware features (simulated)
    current_config_id: int | None
    cache_loaded_configs: list[int]

    # SLO / context
    energy_budget_remaining: float  # 0.0 to 1.0
    latency_target_ms: float

    def to_feature_vector(self) -> list[float]:
        """Convert to numeric feature vector for the RL agent.

        Encodes categorical fields as simple indices, normalizes
        continuous values to roughly [0, 1] range.
        """
        op_encoding = {
            "Conv2d": 0.0, "Linear": 0.25, "MatMul": 0.5,
            "BatchNorm": 0.75, "ReLU": 1.0,
        }.get(self.op_type, 0.5)

        workload_encoding = {
            "compute_bound": 1.0, "memory_bound": 0.0, "balanced": 0.5,
        }.get(self.workload_class, 0.5)

        # Normalize flops (log scale, cap at 1e12)
        import math
        flops_norm = min(1.0, math.log2(self.flops + 1) / 40.0) if self.flops > 0 else 0.0

        # Normalize memory_bytes (log scale, cap at 1e12)
        mem_norm = min(1.0, math.log2(self.memory_bytes + 1) / 40.0) if self.memory_bytes > 0 else 0.0

        # AI normalized (0-50 range mapped to 0-1)
        ai_norm = min(1.0, self.arithmetic_intensity / 50.0)

        # Current config (one-hot-ish, normalized)
        current_cfg = float(self.current_config_id) / 4.0 if self.current_config_id is not None else -1.0

        return [
            op_encoding,
            flops_norm,
            mem_norm,
            ai_norm,
            workload_encoding,
            current_cfg,
            self.energy_budget_remaining,
            min(1.0, self.latency_target_ms / 100.0),
        ]
