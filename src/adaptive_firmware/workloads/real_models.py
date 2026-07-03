"""Real-world workload models for benchmarking.

These represent the three workload patterns the adaptive firmware layer
should handle in production:

1. LLM token streaming: autoregressive generation with growing KV cache.
   Each new token triggers an attention layer that is severely memory-bound
   (large KV cache reads, small compute). The prefill phase is compute-bound
   (parallel token processing).

2. CV detection (YOLO-style): convolutional backbone (compute-bound) feeding
   a detection head with memory-bound operations (large feature map reads).

3. Whisper encoder: mixed conv blocks (compute-bound early layers with small
   feature maps) + attention blocks (memory-bound late layers with large
   context).

All models are small enough to run on CPU on a MacBook. The goal is to
generate realistic workload traces, not to train production models.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..observation.telemetry import WorkloadTrace
from ..hardware.simulator import HardwareSimulator


# ---------------------------------------------------------------------------
# Model 1: LLM token streaming (GPT-2 style)
# ---------------------------------------------------------------------------

class MiniLLMBlock(nn.Module):
    """A small transformer block: attention + FFN.

    Attention is the memory-bound part (reads the KV cache).
    FFN is compute-bound (large matmuls).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, ffn_mult: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, ffn_mult * d_model)
        self.ffn2 = nn.Linear(ffn_mult * d_model, d_model)

    def forward(self, x: torch.Tensor, kv_cache: list | None = None) -> torch.Tensor:
        # Attention with KV cache (memory-bound during decode)
        h = self.ln1(x)
        B, T, D = h.shape
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)
            kv_cache[0] = k.detach()
            kv_cache[1] = v.detach()

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, v.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(attn)

        # FFN (compute-bound)
        h = self.ln2(x)
        h = self.ffn1(h)
        h = F.gelu(h)
        h = self.ffn2(h)
        x = x + h
        return x


class MiniLLM(nn.Module):
    """A tiny GPT-2-like model for trace generation.

    Architecture: token embedding + N transformer blocks + LM head.
    The interesting part is the decode loop, where KV cache makes
    attention memory-bound.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MiniLLMBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers

    def forward(self, input_ids: torch.Tensor, kv_caches: list | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Model 2: CV detection (YOLO-style)
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm + SiLU — the basic YOLO block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))


class MiniYOLO(nn.Module):
    """A tiny YOLO-style detector backbone + head.

    Backbone: 5 conv stages, decreasing spatial resolution, increasing channels.
    Head: a few conv layers for detection.

    Backbone layers are compute-bound (high FLOPs, moderate memory).
    Head layers are memory-bound (large feature maps, small compute).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        # Backbone: spatial downsample 32x32 -> 4x4
        self.stage1 = ConvBNAct(in_channels, 32, stride=2)   # 32x32 -> 16x16
        self.stage2 = ConvBNAct(32, 64, stride=2)            # 16x16 -> 8x8
        self.stage3 = ConvBNAct(64, 128, stride=2)           # 8x8 -> 4x4
        self.stage4 = ConvBNAct(128, 256, stride=1)          # 4x4 (no downsample)

        # Detection head: large feature maps, small compute per pixel
        self.head1 = ConvBNAct(256, 128, kernel=1)
        self.head2 = nn.Conv2d(128, num_classes * 5, kernel_size=1)  # 5 = x,y,w,h,conf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head1(x)
        x = self.head2(x)
        return x


# ---------------------------------------------------------------------------
# Model 3: Whisper-style encoder
# ---------------------------------------------------------------------------

class MiniWhisperEncoder(nn.Module):
    """A tiny Whisper-style audio encoder.

    First half: conv blocks (compute-bound, small feature maps)
    Second half: transformer blocks (memory-bound, large context)

    This represents the transition from compute-bound feature extraction
    to memory-bound sequence modeling.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 128,
        n_conv_layers: int = 2,
        n_transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        # Conv frontend: (batch, n_mels, time) -> (batch, time, d_model)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MiniLLMBlock(d_model, n_heads=4, ffn_mult=2) for _ in range(n_transformer_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (batch, n_mels, time)
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        # Reshape to (batch, time, d_model)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x, kv_cache=None)
        return self.ln_post(x)


# ---------------------------------------------------------------------------
# Trace generators
# ---------------------------------------------------------------------------

def _estimate_attention_decode_flops(
    batch: int, n_heads: int, head_dim: int, seq_len: int, kv_cache_len: int
) -> tuple[float, float]:
    """Estimate FLOPs and memory for one decode-step attention.

    FLOPs: Q @ K^T (B*H*S*KV*2) + softmax * V (B*H*S*KV*2)
    Memory: Q (small) + K cache (large, grows with KV) + V cache (large)
    """
    # Attention scores: Q @ K^T
    flops_qk = 2.0 * batch * n_heads * seq_len * kv_cache_len * head_dim
    # Attention @ V
    flops_av = 2.0 * batch * n_heads * seq_len * kv_cache_len * head_dim
    flops = flops_qk + flops_av

    # Memory: KV cache dominates
    kv_bytes = 2 * batch * n_heads * kv_cache_len * head_dim * 4  # K + V
    q_bytes = batch * n_heads * seq_len * head_dim * 4
    attn_bytes = batch * n_heads * seq_len * kv_cache_len * 4  # attention matrix
    memory_bytes = kv_bytes + q_bytes + attn_bytes

    return flops, memory_bytes


def generate_llm_traces(
    n_prefill_tokens: int = 8,
    n_decode_steps: int = 40,
    d_model: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    batch_size: int = 1,
) -> list[WorkloadTrace]:
    """Generate traces for LLM token streaming.

    Phase 1 (prefill): parallel processing of initial tokens -> compute-bound
    Phase 2 (decode): one token at a time, growing KV cache -> memory-bound

    Returns traces for all layers and phases.
    """
    traces: list[WorkloadTrace] = []
    ffn_mult = 4
    head_dim = d_model // n_heads
    ffn_dim = ffn_mult * d_model

    for layer in range(n_layers):
        # --- Prefill phase: compute-bound ---
        # For each prefill token, run attention + FFN
        for tok in range(n_prefill_tokens):
            # Attention over prefill tokens
            attn_flops = 2.0 * batch_size * n_heads * (tok + 1) * d_model * head_dim
            attn_mem = batch_size * (tok + 1) * d_model * 4 * 3  # Q, K, V

            traces.append(WorkloadTrace(
                op_type="Attention",
                flops=attn_flops,
                memory_bytes=attn_mem,
                batch_size=batch_size,
                tensor_shapes=[(batch_size, tok + 1, d_model)],
                arithmetic_intensity=attn_flops / max(attn_mem, 1),
                workload_class=HardwareSimulator.classify_workload(attn_flops, attn_mem),
            ))

            # FFN: compute-bound
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

        # --- Decode phase: memory-bound (KV cache grows) ---
        for step in range(n_decode_steps):
            kv_len = n_prefill_tokens + step
            # Decode attention: 1 query token, growing KV cache
            attn_flops, attn_mem = _estimate_attention_decode_flops(
                batch=batch_size,
                n_heads=n_heads,
                head_dim=head_dim,
                seq_len=1,  # decode: 1 new token
                kv_cache_len=kv_len,
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

            # Decode FFN: still compute-bound (small seq, large FFN dim)
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

    # LM head (memory-bound: large vocab, small hidden)
    head_flops = 2.0 * batch_size * 1 * d_model * 1024  # vocab=1024
    head_mem = (batch_size * 1 * d_model + 1024) * 4
    for _ in range(n_decode_steps):
        traces.append(WorkloadTrace(
            op_type="LMHead",
            flops=head_flops,
            memory_bytes=head_mem,
            batch_size=batch_size,
            tensor_shapes=[(batch_size, 1, d_model)],
            arithmetic_intensity=head_flops / max(head_mem, 1),
            workload_class=HardwareSimulator.classify_workload(head_flops, head_mem),
        ))

    return traces


def generate_yolo_traces(
    batch_size: int = 1,
    input_size: int = 32,
    num_classes: int = 10,
) -> list[WorkloadTrace]:
    """Generate traces for YOLO-style detection.

    Backbone: 4 conv stages, decreasing spatial resolution, compute-bound
    Head: 2 conv layers, large feature maps, memory-bound
    """
    traces: list[WorkloadTrace] = []
    in_channels = 3

    # Backbone stages: (in_ch, out_ch, stride, name)
    stages = [
        (in_channels, 32, 2, "backbone_s1"),
        (32, 64, 2, "backbone_s2"),
        (64, 128, 2, "backbone_s3"),
        (128, 256, 1, "backbone_s4"),
    ]

    cur_size = input_size
    for in_ch, out_ch, stride, name in stages:
        out_size = cur_size // stride
        # Conv2d FLOPs
        flops = 2.0 * batch_size * out_ch * out_size * out_size * in_ch * 3 * 3
        # BatchNorm: small compute
        bn_flops = batch_size * out_ch * out_size * out_size * 2.0
        # Memory: weights + input + output
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

    # Head: 1x1 conv on large feature maps (memory-bound)
    # head1: 256 -> 128, 1x1
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

    # head2: 128 -> num_classes*5, 1x1
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


def generate_whisper_traces(
    n_mels: int = 80,
    time_steps: int = 100,
    d_model: int = 128,
    n_transformer_layers: int = 2,
    batch_size: int = 1,
) -> list[WorkloadTrace]:
    """Generate traces for Whisper-style audio encoder.

    Phase 1: conv frontend (compute-bound, small feature maps)
    Phase 2: transformer blocks (memory-bound, large context)
    """
    traces: list[WorkloadTrace] = []
    ffn_mult = 2
    head_dim = d_model // 4
    ffn_dim = ffn_mult * d_model
    n_heads = 4

    # Conv frontend: 1D convs on mel spectrogram
    # conv1: (B, n_mels, T) -> (B, d_model, T)
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

    # conv2: stride=2, halves time
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

    # Transformer blocks on the time-reduced sequence
    seq_len = time_steps // 2
    for layer in range(n_transformer_layers):
        # Self-attention: O(seq_len^2) memory
        attn_flops = 2.0 * batch_size * n_heads * seq_len * seq_len * head_dim * 2
        attn_mem = (batch_size * n_heads * seq_len * seq_len * 4  # attention matrix
                    + batch_size * seq_len * d_model * 4 * 3)   # Q, K, V
        traces.append(WorkloadTrace(
            op_type=f"Attention_layer{layer}",
            flops=attn_flops,
            memory_bytes=attn_mem,
            batch_size=batch_size,
            tensor_shapes=[(batch_size, seq_len, d_model)],
            arithmetic_intensity=attn_flops / max(attn_mem, 1),
            workload_class=HardwareSimulator.classify_workload(attn_flops, attn_mem),
        ))

        # FFN
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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result from a single workload benchmark."""

    name: str
    n_traces: int
    workload_class_distribution: dict[str, int]
    adaptive_reward: float
    best_static_reward: float
    best_static_name: str
    reward_improvement: float
    adaptive_time_ms: float
    best_static_time_ms: float
    adaptive_energy_mj: float
    best_static_energy_mj: float
    adaptive_cache_hit_rate: float


def run_workload_benchmark(
    name: str,
    traces: list[WorkloadTrace],
    configs=None,
    seed: int = 42,
) -> BenchmarkResult:
    """Run adaptive vs static comparison on a workload.

    Returns a BenchmarkResult with all key metrics.
    """
    from ..hardware.configs import CONFIG_PRESETS
    from ..runtime.middleware import AdaptiveMiddleware

    if configs is None:
        configs = CONFIG_PRESETS

    random.seed(seed)

    # Adaptive
    mw = AdaptiveMiddleware(
        configs=configs,
        cache_capacity=2,
        learning_rate=0.25,
        epsilon_start=0.4,
    )
    adaptive_report = mw.run_episode(traces)

    # Best static
    static_rewards: dict[str, float] = {}
    static_reports: dict = {}
    for config in configs:
        mw_static = AdaptiveMiddleware(configs=configs, cache_capacity=2)
        report = mw_static.run_static_baseline(traces, config.config_id)
        static_rewards[config.name] = report.avg_reward
        static_reports[config.name] = report

    best_static_name = max(static_rewards, key=static_rewards.get)  # type: ignore[arg-type]
    best_static_report = static_reports[best_static_name]

    # Workload class distribution
    class_dist: dict[str, int] = {}
    for t in traces:
        class_dist[t.workload_class] = class_dist.get(t.workload_class, 0) + 1

    return BenchmarkResult(
        name=name,
        n_traces=len(traces),
        workload_class_distribution=class_dist,
        adaptive_reward=adaptive_report.avg_reward,
        best_static_reward=best_static_report.avg_reward,
        best_static_name=best_static_name,
        reward_improvement=adaptive_report.avg_reward - best_static_report.avg_reward,
        adaptive_time_ms=adaptive_report.total_time_ms,
        best_static_time_ms=best_static_report.total_time_ms,
        adaptive_energy_mj=adaptive_report.total_energy_mj,
        best_static_energy_mj=best_static_report.total_energy_mj,
        adaptive_cache_hit_rate=adaptive_report.cache_hit_rate,
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run benchmarks for all three real-world workload types."""
    results: list[BenchmarkResult] = []

    # LLM token streaming
    llm_traces = generate_llm_traces(
        n_prefill_tokens=8,
        n_decode_steps=40,
        d_model=128,
        n_layers=3,
    )
    results.append(run_workload_benchmark("LLM_token_streaming", llm_traces))

    # YOLO detection
    yolo_traces = generate_yolo_traces(batch_size=1, input_size=32, num_classes=10)
    results.append(run_workload_benchmark("YOLO_detection", yolo_traces))

    # Whisper encoder
    whisper_traces = generate_whisper_traces(
        n_mels=80, time_steps=100, d_model=128, n_transformer_layers=2
    )
    results.append(run_workload_benchmark("Whisper_encoder", whisper_traces))

    return results


def run_mixed_production_benchmark(
    n_inferences: int = 200,
    seed: int = 42,
) -> BenchmarkResult:
    """Simulate a production AI serving system.

    Multiple model types serving concurrent requests. This is the
    scenario where the adaptive firmware layer's value is greatest —
    heterogeneous workloads arriving in unpredictable order.

    Workload mix (representative of a production AI platform):
    - 40% LLM decode (memory-bound, long-running)
    - 25% CV detection (compute-bound, short)
    - 20% Whisper inference (mixed, medium)
    - 15% LLM prefill (compute-bound, bursty)
    """
    import random
    rng = random.Random(seed)

    all_traces: list[WorkloadTrace] = []

    # Generate pools of each type
    llm_decode_pool = generate_llm_traces(
        n_prefill_tokens=0, n_decode_steps=10, d_model=128, n_layers=1
    )
    llm_prefill_pool = generate_llm_traces(
        n_prefill_tokens=4, n_decode_steps=0, d_model=128, n_layers=1
    )
    cv_pool = generate_yolo_traces(batch_size=1, input_size=32, num_classes=10)
    whisper_pool = generate_whisper_traces(
        n_mels=80, time_steps=50, d_model=128, n_transformer_layers=1
    )

    pools = {
        "llm_decode": (llm_decode_pool, 0.40),
        "cv":         (cv_pool, 0.25),
        "whisper":    (whisper_pool, 0.20),
        "llm_prefill": (llm_prefill_pool, 0.15),
    }

    pool_names = list(pools.keys())
    pool_weights = [p[1] for p in pools.values()]

    for _ in range(n_inferences):
        chosen = rng.choices(pool_names, weights=pool_weights, k=1)[0]
        pool = pools[chosen][0]
        all_traces.extend(pool)

    return run_workload_benchmark("Mixed_production", all_traces, seed=seed)


def run_full_benchmark_suite() -> list[BenchmarkResult]:
    """Run the complete benchmark suite:
    - Three single-workload benchmarks (LLM, YOLO, Whisper)
    - One mixed production benchmark (the scenario where adaptation shines)
    """
    results = run_all_benchmarks()
    mixed = run_mixed_production_benchmark(n_inferences=200, seed=42)
    results.append(mixed)
    return results
