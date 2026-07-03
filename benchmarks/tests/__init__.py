"""Test package for the benchmark suite.

Importing this package triggers the workload registrations.
"""
from benchmarks.workloads import (
    llm_decode, llm_prefill, cv_detection, audio_encoder, mixed_production,
)
