"""Mixed production workload.

Simulates a production AI serving system running multiple model types
concurrently. This is the scenario where the adaptive firmware layer
is expected to win — heterogeneous workloads arriving in unpredictable
order, long enough to amortize exploration cost.

Workload mix (representative of a production AI platform):
- 40% LLM decode (memory-bound, long-running)
- 25% CV detection (compute-bound, short)
- 20% Audio encoder (mixed, medium)
- 15% LLM prefill (compute-bound, bursty)
"""

from __future__ import annotations

import random

from adaptive_firmware.observation.telemetry import WorkloadTrace
from .base import WorkloadSpec
from .registry import register_workload


def generate_traces(seed: int = 42, n_inferences: int = 100) -> list[WorkloadTrace]:
    """Generate reproducible mixed production traces.

    Uses the other workloads as building blocks, sampled according to
    the production mix.
    """
    from . import llm_decode, llm_prefill, cv_detection, audio_encoder

    rng = random.Random(seed)
    all_traces: list[WorkloadTrace] = []

    pools = {
        "llm_decode": (llm_decode.generate_traces(seed), 0.40),
        "cv": (cv_detection.generate_traces(seed), 0.25),
        "audio": (audio_encoder.generate_traces(seed), 0.20),
        "llm_prefill": (llm_prefill.generate_traces(seed), 0.15),
    }
    pool_names = list(pools.keys())
    pool_weights = [p[1] for p in pools.values()]

    for _ in range(n_inferences):
        chosen = rng.choices(pool_names, weights=pool_weights, k=1)[0]
        all_traces.extend(pools[chosen][0])

    return all_traces


register_workload(WorkloadSpec(
    name="mixed_production",
    version="1.0.0",
    description=(
        "Production AI serving: 40% LLM decode, 25% CV, 20% audio, 15% LLM prefill. "
        "100 inferences mixed concurrently. The scenario where adaptation shines."
    ),
    tags=["production", "multi-model", "heterogeneous", "mixed"],
    workload_class="multi-tenant",
    trace_generator=generate_traces,
    expected_n_traces=7150,  # approximate; 100 inferences × weighted avg pool size
    seed=42,
))
