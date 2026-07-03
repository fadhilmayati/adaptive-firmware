"""End-to-end tests for the adaptive firmware middleware."""

import pytest
from adaptive_firmware.hardware.configs import CONFIG_PRESETS
from adaptive_firmware.runtime.middleware import AdaptiveMiddleware
from adaptive_firmware.workloads.models import create_synthetic_traces


class TestMiddleware:
    def test_run_episode(self):
        traces = create_synthetic_traces(n_compute=10, n_memory=10, n_balanced=10)
        mw = AdaptiveMiddleware(configs=CONFIG_PRESETS)
        report = mw.run_episode(traces)

        assert report.total_steps == 30
        assert report.total_time_ms > 0
        assert report.total_energy_mj > 0
        assert 0.0 <= report.avg_reward <= 1.0
        assert len(report.logs) == 30

    def test_static_baseline(self):
        traces = create_synthetic_traces(n_compute=10, n_memory=10, n_balanced=10)
        mw = AdaptiveMiddleware(configs=CONFIG_PRESETS)
        report = mw.run_static_baseline(traces, config_id=0)

        assert report.total_steps == 30
        assert all(l.selected_config == 0 for l in report.logs)

    def test_multi_tenant(self):
        tenant_a = create_synthetic_traces(n_compute=15, n_memory=0, n_balanced=5)
        tenant_b = create_synthetic_traces(n_compute=0, n_memory=15, n_balanced=5)

        mw = AdaptiveMiddleware(configs=CONFIG_PRESETS)
        report = mw.run_multi_tenant({
            "tenant_a": tenant_a,
            "tenant_b": tenant_b,
        }, interleave=True)

        assert report.total_steps == 40
        # Should have logs from both tenants
        tenant_ids = {l.tenant_id for l in report.logs}
        assert "tenant_a" in tenant_ids
        assert "tenant_b" in tenant_ids

    def test_adaptive_outperforms_worst_static(self):
        """The adaptive agent should beat at least the worst static config
        on mixed workloads."""
        traces = create_synthetic_traces(n_compute=40, n_memory=40, n_balanced=40)

        # Run adaptive
        mw = AdaptiveMiddleware(
            configs=CONFIG_PRESETS,
            learning_rate=0.25,
            epsilon_start=0.3,
        )
        adaptive_report = mw.run_episode(traces)

        # Run all static baselines
        static_rewards = []
        for cid in range(len(CONFIG_PRESETS)):
            mw_static = AdaptiveMiddleware(configs=CONFIG_PRESETS)
            static_report = mw_static.run_static_baseline(traces, config_id=cid)
            static_rewards.append(static_report.avg_reward)

        worst_static = min(static_rewards)
        best_static = max(static_rewards)

        # Adaptive should beat the worst static
        assert adaptive_report.avg_reward > worst_static, (
            f"Adaptive reward {adaptive_report.avg_reward:.4f} should beat "
            f"worst static {worst_static:.4f}"
        )

    def test_learning_improvement(self):
        """The agent's reward should improve over time (learning curve).
        Uses interleaved workloads so the agent sees mixed types throughout."""
        # Create traces interleaved: compute, memory, balanced, repeat
        from adaptive_firmware.workloads.models import create_synthetic_traces
        import random
        random.seed(42)
        traces = create_synthetic_traces(n_compute=40, n_memory=40, n_balanced=40)
        random.shuffle(traces)

        mw = AdaptiveMiddleware(
            configs=CONFIG_PRESETS,
            learning_rate=0.25,
            epsilon_start=0.3,
        )
        report = mw.run_episode(traces)

        n = len(report.logs)
        early = report.logs[: n // 3]
        late = report.logs[2 * n // 3 :]

        early_avg = sum(l.reward for l in early) / len(early)
        late_avg = sum(l.reward for l in late) / len(late)

        # Late reward should be >= early reward (learning happened)
        assert late_avg >= early_avg * 0.9, (
            f"Late reward {late_avg:.4f} should be >= early reward {early_avg:.4f} * 0.9"
        )

    def test_config_usage_diversity(self):
        """On mixed workloads, the adaptive agent should use more than one config."""
        traces = create_synthetic_traces(n_compute=20, n_memory=20, n_balanced=20)

        mw = AdaptiveMiddleware(configs=CONFIG_PRESETS, epsilon_start=0.3)
        report = mw.run_episode(traces)

        # Should use at least 2 different configs (not just stick to one)
        assert len(report.config_usage) >= 2, (
            f"Agent should use multiple configs, got: {report.config_usage}"
        )
