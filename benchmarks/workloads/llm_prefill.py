"""LLM token streaming (prefill phase) workload.

Represents the prefill phase of an LLM: parallel processing of initial
tokens. Each token triggers attention + FFN over the growing prompt.
This phase is compute-bound (parallel ops, small memory).
"""

from __future__ import annotations

import random

from adaptive_firmware.observation.telemetry import WorkloadTrace
from adaptive_firmware.hardware.simulator import HardwareSimulator
from .base import WorkloadSpec
from .registry import register_workload


def generate_traces(seed: int = 42) -> list[WorkloadTrace]:
    """Generate reproducible LLM prefill traces.

    8 prefill tokens processed in parallel, then 1 decode step.
    Each prefill token produces attention + FFN over the growing prompt.
    """
    rng = random.Random(seed)
    traces: list[WorkloadTrace] = []

    d_model = 128
    n_layers = 2
    n_heads = 4
    head_dim = d_model // n_heads
    ffn_dim = 4 * d_model
    batch_size = 1
    n_prefill = 8

    for layer in range(n_layers):
        for tok in range(n_prefill):
            # Attention over all prefill tokens so far
            attn_flops = 2.0 * batch_size * n_heads * (tok + 1) * d_model * head_dim
            attn_mem = batch_size * (tok + 1) * d_model * 4 * 3
            traces.append(WorkloadTrace(
                op_type="Attention",
                flops=attn_flops,
                memory_bytes=attn_mem,
                batch_size=batch_size,
                tensor_shapes=[(batch_size, tok + 1, d_model)],
                arithmetic_intensity=attn_flops / max(attn_mem, 1),
                workload_class=HardwareSimulator.classify_workload(attn_flops, attn_mem),
            ))

            # FFN
            ffn_flops = 2.0 * batch_size * (tok + 1) * d_model * ffn_dim
            ffn_mem = (batch_size * (tok + 1) * d_model + d_model * ffn_dim) * 4
            traces.append(WorkloadTrace(
                op_type="FFN",
                flops=ffn_flops,
                memory_bytes=ffn_mem,
                batch_size=batch_size,
                tensor_shapes=[(batch_size, tok + 1, d_model)],
                arithmetic_intensity=ffn_flops / max(ffn_mem, 1),
                workload_class=HardwareSimulator.classify_workload(ffn_flops, ffn_mem),
            ))

    return traces


register_workload(WorkloadSpec(
    name="llm_prefill",
    version="1.0.0",
    description="LLM token streaming prefill phase: parallel processing, compute-bound",
    tags=["llm", "compute_bound", "prefill", "parallel"],
    workload_class="single-tenant",
    trace_generator=generate_traces,
    expected_n_traces=32,  # 2 layers × 8 tokens × 2 ops
    seed=42,
))
