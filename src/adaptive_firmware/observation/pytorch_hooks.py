"""PyTorch observer: intercepts forward passes to generate workload traces.

This is the "observation layer" that sits between the AI runtime (PyTorch)
and the adaptive firmware. It uses forward hooks on model layers to capture
operator type, tensor shapes, and estimated FLOPs/memory — the data the RL
agent needs to decide which hardware config to load.

FLOP estimation is approximate (using standard formulas for Conv2d and Linear).
For a production system this would use a proper profiler (torch.profiler),
but for the PoC the hook-based approach is sufficient and much faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .telemetry import WorkloadTrace
from ..hardware.simulator import HardwareSimulator


# FLOP estimation per op type
def _estimate_conv2d_flops(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int],
    input_shape: tuple[int, ...],
) -> tuple[float, float]:
    """Estimate FLOPs and memory for a Conv2d layer.

    FLOPs = 2 * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w
    Memory = weight bytes + input bytes + output bytes (float32)
    """
    batch = input_shape[0]
    in_h, in_w = input_shape[2], input_shape[3]
    kh, kw = kernel_size

    # Output dimensions (stride=1, padding=same assumption)
    out_h = in_h
    out_w = in_w

    flops = 2.0 * batch * out_channels * out_h * out_w * in_channels * kh * kw

    weight_bytes = out_channels * in_channels * kh * kw * 4  # float32
    input_bytes = batch * in_channels * in_h * in_w * 4
    output_bytes = batch * out_channels * out_h * out_w * 4
    memory_bytes = weight_bytes + input_bytes + output_bytes

    return flops, memory_bytes


def _estimate_linear_flops(
    in_features: int,
    out_features: int,
    input_shape: tuple[int, ...],
) -> tuple[float, float]:
    """Estimate FLOPs and memory for a Linear layer."""
    batch = input_shape[0] if len(input_shape) > 1 else 1
    flops = 2.0 * batch * in_features * out_features

    weight_bytes = in_features * out_features * 4  # float32
    input_bytes = batch * in_features * 4
    output_bytes = batch * out_features * 4
    memory_bytes = weight_bytes + input_bytes + output_bytes

    return flops, memory_bytes


def _estimate_matmul_flops(
    shape_a: tuple[int, ...],
    shape_b: tuple[int, ...],
) -> tuple[float, float]:
    """Estimate FLOPs and memory for a MatMul / batched matmul."""
    if len(shape_a) >= 2 and len(shape_b) >= 2:
        m, k = shape_a[-2], shape_a[-1]
        k2, n = shape_b[-2], shape_b[-1]
        batch_dims = 1
        for d in shape_a[:-2]:
            batch_dims *= d
        flops = 2.0 * batch_dims * m * k * n
        memory_bytes = (m * k + k * n + m * n) * batch_dims * 4
        return flops, memory_bytes
    return 0.0, 0.0


class PyTorchObserver:
    """Attaches forward hooks to a PyTorch model and collects workload traces.

    Usage:
        observer = PyTorchObserver(model)
        traces = observer.get_traces()  # after model(x) calls
        observer.clear_traces()
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.traces: list[WorkloadTrace] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on all leaf modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
                handle = module.register_forward_hook(self._make_hook(name, module))
                self._hooks.append(handle)

    def _make_hook(self, name: str, module: nn.Module):
        """Create a forward hook closure for a specific module."""

        def hook_fn(mod: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            flops = 0.0
            memory_bytes = 0.0
            op_type = type(mod).__name__

            input_shape = inp[0].shape if inp and inp[0] is not None else (1,)

            if isinstance(mod, nn.Conv2d):
                flops, memory_bytes = _estimate_conv2d_flops(
                    mod.in_channels,
                    mod.out_channels,
                    mod.kernel_size,
                    tuple(input_shape),
                )
            elif isinstance(mod, nn.Linear):
                flops, memory_bytes = _estimate_linear_flops(
                    mod.in_features,
                    mod.out_features,
                    tuple(input_shape),
                )

            batch_size = input_shape[0] if len(input_shape) > 0 else 1
            ai = flops / memory_bytes if memory_bytes > 0 else float("inf")
            workload_class = HardwareSimulator.classify_workload(flops, memory_bytes)

            self.traces.append(
                WorkloadTrace(
                    op_type=op_type,
                    flops=flops,
                    memory_bytes=memory_bytes,
                    batch_size=int(batch_size),
                    tensor_shapes=[tuple(input_shape)],
                    arithmetic_intensity=ai,
                    workload_class=workload_class,
                )
            )

        return hook_fn

    def get_traces(self) -> list[WorkloadTrace]:
        """Return collected traces (does not clear)."""
        return self.traces

    def clear_traces(self) -> None:
        """Clear collected traces."""
        self.traces.clear()

    def pop_traces(self) -> list[WorkloadTrace]:
        """Return and clear traces (pop pattern)."""
        traces = list(self.traces)
        self.traces.clear()
        return traces

    def remove_hooks(self) -> None:
        """Remove all hooks from the model."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
