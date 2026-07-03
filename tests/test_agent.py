"""Tests for the RL agent."""

import pytest
import numpy as np
from adaptive_firmware.agent.rl_agent import ReconfigAgent
from adaptive_firmware.agent.drift_detector import DriftDetector
from adaptive_firmware.hardware.configs import CONFIG_PRESETS
from adaptive_firmware.observation.telemetry import TelemetryVector


class TestDriftDetector:
    def test_no_drift_with_stable_rewards(self):
        detector = DriftDetector(window_size=20, min_samples=10)
        for _ in range(20):
            drift = detector.update(0.5)
        assert not drift
        assert detector.drift_count == 0

    def test_detects_drift(self):
        detector = DriftDetector(window_size=20, min_samples=10, threshold=1.0)
        # Stable rewards
        for _ in range(10):
            detector.update(0.8)
        # Distribution shifts
        drifts = []
        for _ in range(10):
            drifts.append(detector.update(0.1))
        assert any(drifts)
        assert detector.drift_count > 0

    def test_reset(self):
        detector = DriftDetector()
        detector.update(0.5)
        detector.reset()
        assert len(detector.window) == 0
        assert detector.drift_count == 0


class TestReconfigAgent:
    def _make_telemetry(self, workload_class: str = "compute_bound") -> TelemetryVector:
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

    def test_select_action_returns_valid_config(self):
        agent = ReconfigAgent(CONFIG_PRESETS)
        telemetry = self._make_telemetry()
        action = agent.select_action(telemetry)
        assert 0 <= action < len(CONFIG_PRESETS)

    def test_q_values_update(self):
        agent = ReconfigAgent(CONFIG_PRESETS, learning_rate=0.5, epsilon_start=0.0)
        telemetry = self._make_telemetry("compute_bound")
        # Force a specific action
        action = 0
        initial_q = agent.q_table["compute_bound"][action]
        agent.update(telemetry, action, 0.9)
        assert agent.q_table["compute_bound"][action] != initial_q
        assert agent.q_table["compute_bound"][action] > initial_q

    def test_epsilon_decay(self):
        agent = ReconfigAgent(
            CONFIG_PRESETS,
            epsilon_start=0.5,
            epsilon_end=0.01,
            epsilon_decay=0.9,
        )
        telemetry = self._make_telemetry()
        for _ in range(10):
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.5)
        assert agent.epsilon < 0.5
        assert agent.epsilon >= 0.01

    def test_agent_learns_optimal_config(self):
        """After enough steps, the agent should prefer the optimal config
        for each workload class."""
        np.random.seed(42)
        agent = ReconfigAgent(
            CONFIG_PRESETS,
            learning_rate=0.3,
            epsilon_start=0.5,
            epsilon_end=0.01,
            epsilon_decay=0.97,
        )

        # Simulate many steps with known rewards
        # compute_bound -> config 0 (HIGH_COMPUTE) is best
        for _ in range(200):
            telemetry = self._make_telemetry("compute_bound")
            action = agent.select_action(telemetry)
            # Give high reward for config 0, low for others
            reward = 0.9 if action == 0 else 0.3
            agent.update(telemetry, action, reward)

        policy = agent.get_policy()
        assert policy["compute_bound"] == 0  # Should have learned config 0 is best

    def test_compute_reward_bounds(self):
        agent = ReconfigAgent(CONFIG_PRESETS)

        # Create a mock execution result
        from adaptive_firmware.hardware.simulator import ExecutionResult
        result = ExecutionResult(
            exec_time_ms=10.0,
            energy_mj=5.0,
            reconfig_time_ms=0.0,
            total_time_ms=10.0,
            throughput_gops=500.0,
            config_id=0,
            cache_hit=True,
        )
        reward = agent.compute_reward(result)
        assert 0.0 <= reward <= 1.0
        assert reward > 0.0  # Should be positive for a decent result

    def test_reset(self):
        agent = ReconfigAgent(CONFIG_PRESETS, epsilon_start=0.4)
        telemetry = self._make_telemetry()
        agent.select_action(telemetry)
        agent.update(telemetry, 0, 0.5)
        agent.reset()
        assert agent.stats.total_steps == 0
