"""Deeper analysis suite for sharpening the adaptive firmware thesis."""

from .engine import (
    generate_synthetic_workload,
    scale_reconfig_time,
    run_agent_on_traces,
    run_all_agents_on_traces,
    compute_heterogeneity,
    compute_oracle_gap,
    load_existing_results,
)
