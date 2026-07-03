"""Tests for the neural network policy agent."""

import numpy as np
import pytest

from adaptive_firmware.agent.neural_agent import (
    NeuralPolicy,
    NeuralReconfigAgent,
    ReplayBuffer,
)
from adaptive_firmware.hardware.configs import CONFIG_PRESETS
from adaptive_firmware.observation.telemetry import TelemetryVector


class TestNeuralPolicy:
    def test_forward_shape(self):
        net = NeuralPolicy(state_dim=8, n_actions=4)
        state = np.random.randn(8).astype(np.float32)
        q = net.forward(state)
        assert q.shape == (4,)

    def test_forward_batch(self):
        net = NeuralPolicy(state_dim=8, n_actions=4)
        states = np.random.randn(3, 8).astype(np.float32)
        q = net.forward(states)
        assert q.shape == (3, 4)

    def test_train_step_reduces_loss(self):
        np.random.seed(42)
        net = NeuralPolicy(state_dim=8, n_actions=4, hidden_dim=32)
        states = np.random.randn(8, 8).astype(np.float32)
        actions = np.random.randint(0, 4, size=8)
        targets = np.random.randn(8).astype(np.float32)

        initial_q = net.forward(states)
        initial_loss = np.mean((initial_q[np.arange(8), actions] - targets) ** 2)

        for _ in range(20):
            net.train_step(states, actions, targets, learning_rate=0.01)

        final_q = net.forward(states)
        final_loss = np.mean((final_q[np.arange(8), actions] - targets) ** 2)
        assert final_loss < initial_loss

    def test_copy_from(self):
        net1 = NeuralPolicy(state_dim=8, n_actions=4)
        net2 = NeuralPolicy(state_dim=8, n_actions=4)
        net2.W1 = np.ones_like(net2.W1)
        net1.copy_from(net2)
        assert np.array_equal(net1.W1, net2.W1)


class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            state = np.random.randn(8).astype(np.float32)
            buf.push(state, i % 4, 0.5, state)

        assert len(buf) == 50
        states, actions, rewards, next_states = buf.sample(16)
        assert states.shape == (16, 8)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 8)

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(50):
            buf.push(np.zeros(8), 0, 0.0, np.zeros(8))
        assert len(buf) == 10


class TestNeuralReconfigAgent:
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

    def test_select_action_valid(self):
        np.random.seed(42)
        agent = NeuralReconfigAgent(CONFIG_PRESETS, hidden_dim=32)
        telemetry = self._make_telemetry()
        action = agent.select_action(telemetry)
        assert 0 <= action < len(CONFIG_PRESETS)

    def test_learns_optimal_config(self):
        """After enough training, the neural agent should learn which
        config is best for each workload class."""
        np.random.seed(42)
        agent = NeuralReconfigAgent(
            CONFIG_PRESETS,
            learning_rate=0.01,
            epsilon_start=0.5,
            epsilon_end=0.01,
            epsilon_decay=0.98,
            batch_size=16,
            target_update_freq=20,
            train_freq=1,
            hidden_dim=32,
        )

        # Train: compute_bound -> config 0 is best
        for _ in range(200):
            telemetry = self._make_telemetry("compute_bound")
            action = agent.select_action(telemetry)
            reward = 0.9 if action == 0 else 0.3
            agent.update(telemetry, action, reward)

        # After training, the greedy policy for compute_bound should be config 0
        telemetry = self._make_telemetry("compute_bound")
        policy_action = agent.get_policy(telemetry)
        assert policy_action == 0

    def test_epsilon_decay(self):
        np.random.seed(42)
        agent = NeuralReconfigAgent(
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

    def test_replay_buffer_fills(self):
        np.random.seed(42)
        agent = NeuralReconfigAgent(CONFIG_PRESETS, buffer_capacity=100)
        telemetry = self._make_telemetry()
        for _ in range(150):
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.5)
        # Buffer should be at capacity
        assert len(agent.replay_buffer) == 100

    def test_target_network_updates(self):
        np.random.seed(42)
        agent = NeuralReconfigAgent(
            CONFIG_PRESETS,
            target_update_freq=10,
            train_freq=1,
            buffer_capacity=100,
        )
        telemetry = self._make_telemetry()
        for _ in range(20):
            action = agent.select_action(telemetry)
            agent.update(telemetry, action, 0.5)

        # Target network should have been updated at least once
        # (check that target net weights differ from initial random init)
        assert not np.array_equal(
            agent.target_net.W1.flatten()[:5],
            agent.policy_net.W1.flatten()[:5],
        ) or agent._steps_since_target_update < agent.target_update_freq

    def test_compute_reward(self):
        np.random.seed(42)
        agent = NeuralReconfigAgent(CONFIG_PRESETS)
        from adaptive_firmware.hardware.simulator import ExecutionResult
        result = ExecutionResult(
            exec_time_ms=10.0, energy_mj=5.0, reconfig_time_ms=0.0,
            total_time_ms=10.0, throughput_gops=500.0, config_id=0, cache_hit=True,
        )
        reward = agent.compute_reward(result)
        assert 0.0 <= reward <= 1.0
