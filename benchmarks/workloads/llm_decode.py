"""LLM token streaming (decode phase) workload.

Represents the autoregressive decode phase of an LLM: one new token at
a time, growing KV cache, attention layers that become increasingly
memory-bound as the cache grows.

This is the workload where the adaptive firmware layer is most likely
to win — the decode phase is long-running, the workload is
temporally homogeneous (mostly memory-bound), and the static
LOW_POWER config is competitive but not optimal.
"""

from __future__ import annotations

import math
import random

from adaptive_firmware.observation.telemetry import WorkloadTrace
from adaptive_firmware.hardware.simulator import HardwareSimulator
from .base import WorkloadSpec
from .registry import register_workload


def _estimate_attention_decode_flops(
    batch: int, n_heads: int, head_dim: int, seq_len: int, kv_cache_len: int
) -> tuple[float, float]:
    """Estimate FLOPs and memory for one decode-step attention."""
    flops_qk = 2.0 * batch * n_heads * seq_len * kv_cache_len * head_dim
    flops_av = 2.0 * batch * n_heads * seq_len * kv_cache_len * head_dim
    flops = flops_qk + flops_av
    kv_bytes = 2 * batch * n_heads * kv_cache_len * head_dim * 4
    q_bytes = batch * n_heads * seq_len * head_dim * 4
    attn_bytes = batch * n_heads * seq_len * kv_cache_len * 4
    memory_bytes = kv_bytes + q_bytes + attn_bytes
    return flops, memory_bytes


def generate_traces(seed: int = 42) -> list[WorkloadTrace]:
    """Generate reproducible LLM decode traces.

    40 decode steps with growing KV cache. Each step produces
    attention (memory-bound, KV cache grows) + FFN (compute-bound, small seq).
    """
    rng = random.Random(seed)
    traces: list[WorkloadTrace] = []

    d_model = 128
    n_layers = 2
    n_heads = 4
    head_dim = d_model // n_heads
    ffn_dim = 4 * d_model
    batch_size = 1

    for layer in range(n_layers):
        for step in range(40):
            kv_len = 8 + step  # Growing KV cache
            attn_flops, attn_mem = _estimate_attention_decode_flops(
                batch=batch_size, n_heads=n_heads, head_dim=head_dim,
                seq_len=1, kv_cache_len=kv_len,
            )
            traces.append(WorkloadTrace(
                op_type="Attention",
                flops=attn_flops,
                memory_bytes=attn_mem,
                batch_size=batch_size,
                tensor_shapes=[(batch_size, 1, d_model)],
                arithmetic_intensity=attn_flops / max(attn_mem, 1),
                workload_class=HardwareSimulator.classify_workload(attn_flops, attn_mem),
            ))

            ffn_flops = 2.0 * batch_size * 1 * d_model * ffn_dim
            ffn_mem = (batch_size * 1 * d_model + d_model * ffn_dim) * 4
            traces.append(WorkloadTrace(
                op_type="FFN",
                flops=ffn_flops,
                memory_bytes=ffn_mem,
                batch_size=batch_size,
                tensor_shapes=[(batch_size, 1, d_model)],
                arithmetic_intensity=ffn_flops / max(ffn_mem, 1),
                workload_class=HardwareSimulator.classify_workload(ffn_flops, ffn_mem),
            ))

    return traces


register_workload(WorkloadSpec(
    name="llm_decode",
    version="1.0.0",
    description="LLM token streaming decode phase: growing KV cache, memory-bound attention",
    tags=["llm", "memory_bound", "decode", "autoregressive"],
    workload_class="single-tenant",
    trace_generator=generate_traces,
    expected_n_traces=160,  # 2 layers × 40 steps × 2 ops
    seed=42,
))
