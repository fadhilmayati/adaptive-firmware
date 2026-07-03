"""Pytest configuration for the benchmark suite.

Ensures all workloads are registered before any test runs by importing
the workloads package, which triggers the side-effect registrations
in each workload module.
"""

# Importing these modules triggers the register_workload() calls
# at module level. This must happen before any test that uses
# the workload registry.
from benchmarks.workloads import (
    llm_decode,
    llm_prefill,
    cv_detection,
    audio_encoder,
    mixed_production,
)
