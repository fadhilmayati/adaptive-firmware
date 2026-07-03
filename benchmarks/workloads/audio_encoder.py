"""Audio encoder (Whisper-style) workload.

Represents an audio encoder: conv frontend (compute-bound) followed by
transformer blocks (memory-bound). Short and mostly homogeneous.
"""

from __future__ import annotations

import random

from adaptive_firmware.observation.telemetry import WorkloadTrace
from adaptive_firmware.hardware.simulator import HardwareSimulator
from .base import WorkloadSpec
from .registry import register_workload


def generate_traces(seed: int = 42) -> list[WorkloadTrace]:
    """Generate reproducible Whisper-style encoder traces."""
    rng = random.Random(seed)
    traces: list[WorkloadTrace] = []

    n_mels = 80
    time_steps = 100
    d_model = 128
    n_transformer_layers = 2
    batch_size = 1
    n_heads = 4
    head_dim = d_model // n_heads
    ffn_dim = 2 * d_model

    # Conv frontend
    conv1_flops = 2.0 * batch_size * d_model * time_steps * n_mels * 3
    conv1_mem = (d_model * n_mels * 3 + batch_size * n_mels * time_steps
                 + batch_size * d_model * time_steps) * 4
    traces.append(WorkloadTrace(
        op_type="Conv1d_frontend1",
        flops=conv1_flops,
        memory_bytes=conv1_mem,
        batch_size=batch_size,
        tensor_shapes=[(batch_size, n_mels, time_steps)],
        arithmetic_intensity=conv1_flops / max(conv1_mem, 1),
        workload_class=HardwareSimulator.classify_workload(conv1_flops, conv1_mem),
    ))

    conv2_flops = 2.0 * batch_size * d_model * (time_steps // 2) * d_model * 3
    conv2_mem = (d_model * d_model * 3 + batch_size * d_model * time_steps
                 + batch_size * d_model * (time_steps // 2)) * 4
    traces.append(WorkloadTrace(
        op_type="Conv1d_frontend2",
        flops=conv2_flops,
        memory_bytes=conv2_mem,
        batch_size=batch_size,
        tensor_shapes=[(batch_size, d_model, time_steps // 2)],
        arithmetic_intensity=conv2_flops / max(conv2_mem, 1),
        workload_class=HardwareSimulator.classify_workload(conv2_flops, conv2_mem),
    ))

    # Transformer blocks
    seq_len = time_steps // 2
    for layer in range(n_transformer_layers):
        attn_flops = 2.0 * batch_size * n_heads * seq_len * seq_len * head_dim * 2
        attn_mem = (batch_size * n_heads * seq_len * seq_len * 4
                    + batch_size * seq_len * d_model * 4 * 3)
        traces.append(WorkloadTrace(
            op_type=f"Attention_layer{layer}",
            flops=attn_flops,
            memory_bytes=attn_mem,
            batch_size=batch_size,
            tensor_shapes=[(batch_size, seq_len, d_model)],
            arithmetic_intensity=attn_flops / max(attn_mem, 1),
            workload_class=HardwareSimulator.classify_workload(attn_flops, attn_mem),
        ))

        ffn_flops = 2.0 * batch_size * seq_len * d_model * ffn_dim
        ffn_mem = (batch_size * seq_len * d_model + d_model * ffn_dim) * 4 * 2
        traces.append(WorkloadTrace(
            op_type=f"FFN_layer{layer}",
            flops=ffn_flops,
            memory_bytes=ffn_mem,
            batch_size=batch_size,
            tensor_shapes=[(batch_size, seq_len, d_model)],
            arithmetic_intensity=ffn_flops / max(ffn_mem, 1),
            workload_class=HardwareSimulator.classify_workload(ffn_flops, ffn_mem),
        ))

    return traces


register_workload(WorkloadSpec(
    name="audio_encoder",
    version="1.0.0",
    description="Whisper-style audio encoder: compute-bound conv frontend + memory-bound transformer",
    tags=["audio", "whisper", "encoder", "mixed"],
    workload_class="single-tenant",
    trace_generator=generate_traces,
    expected_n_traces=6,  # 2 conv + 2 attn + 2 ffn
    seed=42,
))
