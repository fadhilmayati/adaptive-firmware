"""Tests for the benchmark suite itself."""

import json
import pytest
from pathlib import Path

# Importing these modules triggers the register_workload() side effects
# at module level. Must happen before any test uses the registry.
from benchmarks.workloads import (
    llm_decode, llm_prefill, cv_detection, audio_encoder, mixed_production,
)
from benchmarks.workloads.base import WorkloadSpec
from benchmarks.workloads.registry import (
    list_workloads, get_workload, register_workload, clear_registry
)
from benchmarks.runner import BenchmarkRunner, run_workload
from benchmarks.aggregator import load_results, aggregate_results, generate_leaderboard


class TestWorkloadRegistry:
    def test_all_workloads_registered(self):
        """All five canonical workloads should be registered after import."""
        from benchmarks.workloads import llm_decode, llm_prefill, cv_detection, audio_encoder, mixed_production
        workloads = list_workloads()
        names = {w.name for w in workloads}
        assert "llm_decode" in names
        assert "llm_prefill" in names
        assert "cv_detection" in names
        assert "audio_encoder" in names
        assert "mixed_production" in names

    def test_get_workload_by_name(self):
        from benchmarks.workloads import llm_decode
        spec = get_workload("llm_decode")
        assert spec.name == "llm_decode"
        assert spec.version == "1.0.0"

    def test_get_workload_by_name_and_version(self):
        from benchmarks.workloads import llm_decode
        spec = get_workload("llm_decode", "1.0.0")
        assert spec.name == "llm_decode"
        assert spec.version == "1.0.0"

    def test_get_workload_unknown_raises(self):
        with pytest.raises(KeyError):
            get_workload("nonexistent_workload")

    def test_register_duplicate_raises(self):
        """Registering a workload with an already-used (name, version) raises."""
        spec = WorkloadSpec(
            name="new_test_workload",
            version="1.0.0",
            description="test",
            tags=[],
            workload_class="single-tenant",
            trace_generator=lambda s: [],
            expected_n_traces=0,
        )
        register_workload(spec)
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_workload(spec)
        finally:
            # Clean up: remove the test workload from the registry
            from benchmarks.workloads.registry import _REGISTRY
            _REGISTRY.pop("new_test_workload@1.0.0", None)

    def test_workload_generate_is_deterministic(self):
        """Same seed → same traces."""
        from benchmarks.workloads import llm_decode
        spec = get_workload("llm_decode")
        traces1 = spec.generate()
        traces2 = spec.generate()
        assert len(traces1) == len(traces2)
        assert all(
            t1.flops == t2.flops and t1.memory_bytes == t2.memory_bytes
            for t1, t2 in zip(traces1, traces2)
        )

    def test_workload_validate(self):
        from benchmarks.workloads import llm_decode
        spec = get_workload("llm_decode")
        traces = spec.generate()
        assert spec.validate(traces)


class TestBenchmarkRunner:
    def test_run_oracle(self, tmp_path):
        """Oracle should always get the highest reward (by definition)."""
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        result = runner.run("cv_detection", agent="oracle")
        assert result.agent_name == "oracle"
        assert result.n_traces == 6
        assert result.avg_reward > 0.0

    def test_run_static(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        result = runner.run("cv_detection", agent="static_2")
        assert result.agent_name == "static_2"
        assert result.n_traces == 6

    def test_run_tabular(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        result = runner.run("llm_decode", agent="tabular")
        assert result.agent_name == "tabular"
        assert result.n_traces == 160  # 2 layers × 40 steps × 2 ops

    def test_run_random(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        result = runner.run("cv_detection", agent="random", agent_config={"seed": 42})
        assert result.agent_name == "random"

    def test_run_unknown_agent_raises(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown agent"):
            runner.run("cv_detection", agent="nonexistent_agent")

    def test_run_writes_json(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        result = runner.run("cv_detection", agent="oracle")
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["workload_name"] == "cv_detection"
        assert data["agent_name"] == "oracle"

    def test_result_to_json(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        result = runner.run("cv_detection", agent="oracle")
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["suite_version"] == "0.1.0"
        assert "timestamp" in parsed


class TestAggregator:
    def test_load_results_empty(self, tmp_path):
        results = load_results(str(tmp_path))
        assert results == []

    def test_aggregate_results(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        runner.run("cv_detection", agent="oracle")
        runner.run("cv_detection", agent="static_2")

        results = load_results(str(tmp_path))
        agg = aggregate_results(results)

        assert "cv_detection" in agg["by_workload"]
        assert "oracle" in agg["by_agent"]
        assert "cv_detection" in agg["best_per_workload"]
        assert len(agg["leaderboard"]) == 2

    def test_generate_leaderboard(self, tmp_path):
        runner = BenchmarkRunner(output_dir=str(tmp_path))
        runner.run("cv_detection", agent="oracle")
        runner.run("cv_detection", agent="static_2")

        leaderboard = generate_leaderboard(results_dir=str(tmp_path))
        assert "Benchmark Leaderboard" in leaderboard
        assert "cv_detection" in leaderboard
        assert "oracle" in leaderboard
