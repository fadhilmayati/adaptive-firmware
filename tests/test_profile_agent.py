"""Tests for the ProfileThenCommitAgent."""

import numpy as np
import pytest

from adaptive_firmware.agent.profile_agent import ProfileThenCommitAgent
from adaptive_firmware.hardware.configs import CONFIG_PRESETS
from adaptive_firmware.observation.telemetry import TelemetryVector


def _make_telemetry(workload_class: str = "compute_bound") -> TelemetryVector:
    return TelemetryVector(
        op_type="Conv2d",
        flops=1e9,
        memory_bytes=1e7,
        arithmetic_intensity=100.0,
        workload_class=workload_class,
        current_config_id=0,
        cache_loaded_configs=[0],
        energy_budget_remaining=1.0,
        latency_target_ms=50.0,
    )


class TestProfileThenCommitAgent:
    def test_starts_in_profile_phase(self):
        np.random.seed(42)
        agent = ProfileThenCommitAgent(CONFIG_PRESETS, profile_steps=5)
        assert agent._phase == "profile"
        assert agent.profile_complete is False

    def test_profiles_each_config(self):
        np.random.seed(42)
        agent = ProfileThenCommitAgent(CONFIG_PRESETS, profile_steps=3)
        # Run 3*4 = 12 steps; the 13th triggers the commit
        actions = []
        for _ in range(12):
            actions.append(agent.select_action(_make_telemetry()))
        # Should have visited each config 3 times during profiling
        from collections import Counter
        counts = Counter(actions)
        assert len(counts) == 4  # all 4 configs visited
        assert all(c == 3 for c in counts.values())

    def test_commits_to_best_config(self):
        """After profiling, agent should commit to the config with highest avg reward."""
        np.random.seed(42)
        # Use epsilon=0 to eliminate exploration during commit
        agent = ProfileThenCommitAgent(
            CONFIG_PRESETS, profile_steps=3, commit_epsilon=0.0
        )
        # Run 13 steps: 12 profile + 1 to trigger commit
        for _ in range(13):
            telemetry = _make_telemetry()
            action = agent.select_action(telemetry)
            reward = 0.9 if action == 0 else 0.3
            agent.update(telemetry, action, reward)
        assert agent._phase == "commit"
        assert agent._committed_config == 0
        # All subsequent selections should be config 0 (no exploration)
        for _ in range(10):
            assert agent.select_action(_make_telemetry()) == 0

    def test_commit_with_exploration(self):
        """With commit_epsilon=0, no exploration. With epsilon=1, always explore."""
        np.random.seed(42)
        agent_no_explore = ProfileThenCommitAgent(
            CONFIG_PRESETS, profile_steps=3, commit_epsilon=0.0
        )
        for _ in range(13):
            t = _make_telemetry()
            a = agent_no_explore.select_action(t)
            agent_no_explore.update(t, a, 0.9 if a == 0 else 0.3)
        # All selections should be committed config
        selections = [agent_no_explore.select_action(_make_telemetry()) for _ in range(50)]
        assert all(s == 0 for s in selections)

    def test_profile_phase_cost_is_bounded(self):
        """Profile phase should be exactly n_configs * profile_steps calls."""
        np.random.seed(42)
        profile_steps = 5
        agent = ProfileThenCommitAgent(CONFIG_PRESETS, profile_steps=profile_steps)
        n_profile_calls = 0
        for _ in range(100):
            telemetry = _make_telemetry()
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.5)
            if agent._phase == "profile":
                n_profile_calls += 1
        expected = len(CONFIG_PRESETS) * profile_steps
        assert n_profile_calls == expected

    def test_reprofiles_on_drift(self):
        """Drift detection should trigger re-profiling."""
        np.random.seed(42)
        agent = ProfileThenCommitAgent(
            CONFIG_PRESETS,
            profile_steps=3,
            commit_epsilon=0.0,
            drift_window=10,
            drift_threshold=0.5,  # sensitive threshold
        )
        # Complete profile phase
        for _ in range(13):
            telemetry = _make_telemetry()
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.5)
        assert agent._phase == "commit"
        assert agent.reprofiles == 0
        # Inject a sharp reward change — high then low
        for _ in range(5):
            telemetry = _make_telemetry()
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.9)
        for _ in range(5):
            telemetry = _make_telemetry()
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.0)
        # Should have re-profiled at least once
        assert agent.reprofiles >= 1

    def test_compute_reward_bounds(self):
        from adaptive_firmware.hardware.simulator import ExecutionResult
        agent = ProfileThenCommitAgent(CONFIG_PRESETS)
        result = ExecutionResult(
            exec_time_ms=10.0, energy_mj=5.0, reconfig_time_ms=0.0,
            total_time_ms=10.0, throughput_gops=500.0, config_id=0, cache_hit=True,
        )
        reward = agent.compute_reward_for_result(result)
        assert 0.0 <= reward <= 1.0

    def test_reset(self):
        agent = ProfileThenCommitAgent(CONFIG_PRESETS, profile_steps=3)
        for _ in range(20):
            telemetry = _make_telemetry()
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.5)
        agent.reset()
        assert agent._phase == "profile"
        assert agent.profile_complete is False
        assert agent.steps == 0
