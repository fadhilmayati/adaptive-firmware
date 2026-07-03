"""Test workloads for the adaptive firmware PoC.

These models generate workload traces with different characteristics:
- CNN model: produces compute-bound traces (high FLOP, moderate memory)
- Linear/MLP model: produces balanced traces
- Memory-heavy model: produces memory-bound traces (low FLOP, high memory)

The workload sequence cycles through these phases to simulate a real
AI runtime that processes different model types over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..observation.telemetry import WorkloadTrace
from ..hardware.simulator import HardwareSimulator


@dataclass
class WorkloadPhase:
    """A named phase in a workload sequence."""

    name: str
    model: nn.Module
    input_shape: tuple[int, ...]
    n_ops: int  # expected number of ops per forward pass


class SimpleCNN(nn.Module):
    """A small CNN that produces compute-bound workloads."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    """A small MLP that produces balanced workloads."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MemoryHeavyModel(nn.Module):
    """A model with small compute but large tensors (memory-bound).

    Uses large Linear layers with small batch — low arithmetic intensity.
    """

    def __init__(self) -> None:
        super().__init__()
        # Large weight matrices, small input -> memory-bound
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_workload_sequence(
    n_cycles: int = 3,
    batch_size: int = 4,
) -> tuple[list[WorkloadTrace], list[WorkloadPhase]]:
    """Create a mixed workload sequence that cycles through model phases.

    Each cycle runs:
    1. CNN model (compute-bound)
    2. MLP model (balanced)
    3. Memory-heavy model (memory-bound)

    This cycling creates concept drift — the workload distribution changes
    between phases, which the agent must detect and adapt to.

    Args:
        n_cycles: Number of full cycles through all phases.
        batch_size: Batch size for all models.

    Returns:
        Tuple of (traces, phases) where traces is a flat list and phases
        records which model produced which segment.
    """
    from ..observation.pytorch_hooks import PyTorchObserver

    phases = [
        WorkloadPhase(
            name="cnn_compute_bound",
            model=SimpleCNN(),
            input_shape=(batch_size, 3, 32, 32),
            n_ops=4,
        ),
        WorkloadPhase(
            name="mlp_balanced",
            model=SimpleMLP(),
            input_shape=(batch_size, 256),
            n_ops=3,
        ),
        WorkloadPhase(
            name="memory_heavy",
            model=MemoryHeavyModel(),
            input_shape=(1, 4096),  # batch=1 for memory-bound
            n_ops=2,
        ),
    ]

    all_traces: list[WorkloadTrace] = []

    for cycle in range(n_cycles):
        for phase in phases:
            model = phase.model
            model.eval()

            observer = PyTorchObserver(model)

            with torch.no_grad():
                for _ in range(phase.n_ops):
                    inp = torch.randn(*phase.input_shape)
                    model(inp)

            traces = observer.pop_traces()
            all_traces.extend(traces)
            observer.remove_hooks()

    return all_traces, phases


def create_synthetic_traces(
    n_compute: int = 50,
    n_memory: int = 50,
    n_balanced: int = 50,
) -> list[WorkloadTrace]:
    """Create synthetic workload traces without running PyTorch.

    Useful for fast testing and benchmarking. Generates traces with
    known arithmetic intensity profiles.
    """
    traces: list[WorkloadTrace] = []

    # Compute-bound: high FLOPs, low memory (AI > 20)
    for i in range(n_compute):
        flops = 1e9 + i * 1e8  # 1-6 GFLOP
        memory = flops / 30.0  # AI = 30
        traces.append(WorkloadTrace(
            op_type="Conv2d",
            flops=flops,
            memory_bytes=memory,
            batch_size=4,
            tensor_shapes=[(4, 3, 32, 32)],
            arithmetic_intensity=flops / memory,
            workload_class="compute_bound",
        ))

    # Memory-bound: low FLOPs, high memory (AI < 5)
    for i in range(n_memory):
        memory = 1e9 + i * 1e8  # 1-6 GB
        flops = memory * 2.0  # AI = 2
        traces.append(WorkloadTrace(
            op_type="Linear",
            flops=flops,
            memory_bytes=memory,
            batch_size=1,
            tensor_shapes=[(1, 4096)],
            arithmetic_intensity=flops / memory,
            workload_class="memory_bound",
        ))

    # Balanced: moderate FLOPs, moderate memory (5 < AI < 20)
    for i in range(n_balanced):
        flops = 5e8 + i * 5e7
        memory = flops / 10.0  # AI = 10
        traces.append(WorkloadTrace(
            op_type="Linear",
            flops=flops,
            memory_bytes=memory,
            batch_size=4,
            tensor_shapes=[(4, 256)],
            arithmetic_intensity=flops / memory,
            workload_class="balanced",
        ))

    return traces
