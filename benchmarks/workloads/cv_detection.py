"""CV detection (YOLO-style) workload.

Represents an object detection model: convolutional backbone
(compute-bound) + detection head (memory-bound). Short and
homogeneous, so the adaptive firmware layer is expected to lose
to a well-chosen static config.
"""

from __future__ import annotations

import random

from adaptive_firmware.observation.telemetry import WorkloadTrace
from adaptive_firmware.hardware.simulator import HardwareSimulator
from .base import WorkloadSpec
from .registry import register_workload


def generate_traces(seed: int = 42) -> list[WorkloadTrace]:
    """Generate reproducible YOLO-style detection traces."""
    rng = random.Random(seed)
    traces: list[WorkloadTrace] = []

    batch_size = 1
    in_channels = 3
    num_classes = 10

    stages = [
        (in_channels, 32, 2, "s1"),
        (32, 64, 2, "s2"),
        (64, 128, 2, "s3"),
        (128, 256, 1, "s4"),
    ]

    cur_size = 32
    for in_ch, out_ch, stride, name in stages:
        out_size = cur_size // stride
        flops = 2.0 * batch_size * out_ch * out_size * out_size * in_ch * 3 * 3
        bn_flops = batch_size * out_ch * out_size * out_size * 2.0
        weight_bytes = out_ch * in_ch * 3 * 3 * 4
        in_bytes = batch_size * in_ch * cur_size * cur_size * 4
        out_bytes = batch_size * out_ch * out_size * out_size * 4
        memory_bytes = weight_bytes + in_bytes + out_bytes

        total_flops = flops + bn_flops
        traces.append(WorkloadTrace(
            op_type=f"Conv2d_{name}",
            flops=total_flops,
            memory_bytes=memory_bytes,
            batch_size=batch_size,
            tensor_shapes=[(batch_size, in_ch, cur_size, cur_size)],
            arithmetic_intensity=total_flops / max(memory_bytes, 1),
            workload_class=HardwareSimulator.classify_workload(total_flops, memory_bytes),
        ))
        cur_size = out_size

    # Head: 1x1 conv on large feature maps
    flops_h1 = 2.0 * batch_size * 128 * cur_size * cur_size * 256 * 1 * 1
    mem_h1 = (128 * 256 + batch_size * 256 * cur_size * cur_size
              + batch_size * 128 * cur_size * cur_size) * 4
    traces.append(WorkloadTrace(
        op_type="Conv2d_head1",
        flops=flops_h1,
        memory_bytes=mem_h1,
        batch_size=batch_size,
        tensor_shapes=[(batch_size, 256, cur_size, cur_size)],
        arithmetic_intensity=flops_h1 / max(mem_h1, 1),
        workload_class=HardwareSimulator.classify_workload(flops_h1, mem_h1),
    ))

    flops_h2 = 2.0 * batch_size * num_classes * 5 * cur_size * cur_size * 128 * 1 * 1
    mem_h2 = (num_classes * 5 * 128 + batch_size * 128 * cur_size * cur_size
              + batch_size * num_classes * 5 * cur_size * cur_size) * 4
    traces.append(WorkloadTrace(
        op_type="Conv2d_head2",
        flops=flops_h2,
        memory_bytes=mem_h2,
        batch_size=batch_size,
        tensor_shapes=[(batch_size, 128, cur_size, cur_size)],
        arithmetic_intensity=flops_h2 / max(mem_h2, 1),
        workload_class=HardwareSimulator.classify_workload(flops_h2, mem_h2),
    ))

    return traces


register_workload(WorkloadSpec(
    name="cv_detection",
    version="1.0.0",
    description="YOLO-style object detection: compute-bound backbone + memory-bound head",
    tags=["cv", "yolo", "detection", "mixed"],
    workload_class="single-tenant",
    trace_generator=generate_traces,
    expected_n_traces=6,  # 4 backbone + 2 head
    seed=42,
))
