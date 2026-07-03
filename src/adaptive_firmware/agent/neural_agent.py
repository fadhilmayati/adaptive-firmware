"""Neural network policy for the adaptive firmware layer.

Replaces tabular Q-learning with a small MLP that maps telemetry features
to Q-values for each accelerator config. The MLP generalizes across
continuous state spaces and can transfer learning between workload types.

Architecture:
- Input: 8 telemetry features (from TelemetryVector.to_feature_vector())
- Hidden: 2 layers of 64 units with ReLU
- Output: 4 Q-values (one per accelerator config)
- Training: online SGD with experience replay buffer
- Target network: periodic copy for stable Q-learning targets

Why a small MLP, not a transformer or large network?
- The state space is small (8 features)
- The action space is discrete and small (4 configs)
- We need fast inference (the loop budget is 1-15ms)
- We need fast online updates (no batch training)
- A 64-unit MLP runs in microseconds on CPU

The MLP is small enough to run on the ARM Cortex-A53 in a Kria KV260
without dedicated acceleration. That's the deployment target.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from .drift_detector import DriftDetector
from ..observation.telemetry import TelemetryVector
from ..hardware.configs import AcceleratorConfig


@dataclass
class NNAgentStats:
    """Running statistics for the neural network agent."""

    total_steps: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    config_selections: dict[int, int] = None
    exploration_rate: float = 0.0
    best_config_per_state: dict[str, int] = None

    def __post_init__(self):
        if self.config_selections is None:
            self.config_selections = {}
        if self.best_config_per_state is None:
            self.best_config_per_state = {}


class ReplayBuffer:
    """Simple experience replay buffer for online RL.

    Stores (state, action, reward, next_state) tuples and samples
    mini-batches for training. This stabilizes Q-learning by
    breaking the temporal correlation between consecutive updates.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class NeuralPolicy:
    """A small MLP Q-network implemented in pure NumPy.

    We use NumPy instead of PyTorch for the agent because:
    - No framework overhead — fits in the 1-15ms loop budget
    - Easy to inspect, debug, and port to C/Verilog later
    - Runs on any CPU, no GPU required
    - Tiny deployment footprint (no PyTorch runtime dependency)

    For production, this would be ported to a compiled framework
    or hardware accelerator. The NumPy version is the reference.
    """

    def __init__(
        self,
        state_dim: int = 8,
        n_actions: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        """Initialize the MLP with random weights (Xavier init)."""
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        rng = np.random.RandomState(42)

        # Layer 1: state_dim -> hidden_dim
        self.W1 = rng.randn(state_dim, hidden_dim).astype(np.float32) * math.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Layer 2: hidden_dim -> hidden_dim
        self.W2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * math.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        # Output: hidden_dim -> n_actions
        self.W3 = rng.randn(hidden_dim, n_actions).astype(np.float32) * math.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: state -> Q-values for each action.

        Uses ReLU activations in hidden layers, linear output.
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # Layer 1
        h1 = state @ self.W1 + self.b1
        h1 = np.maximum(h1, 0)  # ReLU

        # Layer 2
        h2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(h2, 0)  # ReLU

        # Output
        q_values = h2 @ self.W3 + self.b3
        return q_values.squeeze() if q_values.shape[0] == 1 else q_values

    def copy_from(self, other: "NeuralPolicy") -> None:
        """Copy weights from another network (for target network updates)."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.001,
    ) -> float:
        """One step of SGD on a mini-batch.

        Uses MSE loss between predicted Q-values and targets.
        Returns the loss for monitoring.
        """
        batch_size = states.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            state = states[i:i+1]
            action = int(actions[i])
            target = float(targets[i])

            # Forward pass
            h1 = state @ self.W1 + self.b1
            h1 = np.maximum(h1, 0)
            h2 = h1 @ self.W2 + self.b2
            h2 = np.maximum(h2, 0)
            q_values = h2 @ self.W3 + self.b3

            # Loss
            predicted = q_values[0, action]
            loss = (predicted - target) ** 2
            total_loss += loss

            # Backward pass (gradient of MSE w.r.t. Q-values)
            grad_q = np.zeros(self.n_actions, dtype=np.float32)
            grad_q[action] = 2.0 * (predicted - target)

            # Gradients through W3, b3
            grad_W3 = h2.T @ grad_q.reshape(1, -1)
            grad_b3 = grad_q

            # Gradients through h2
            grad_h2 = grad_q @ self.W3.T
            grad_h2 = grad_h2 * (h2 > 0)  # ReLU derivative

            # Gradients through W2, b2
            grad_W2 = h1.T @ grad_h2
            grad_b2 = grad_h2.squeeze()

            # Gradients through h1
            grad_h1 = grad_h2 @ self.W2.T
            grad_h1 = grad_h1 * (h1 > 0)  # ReLU derivative

            # Gradients through W1, b1
            grad_W1 = state.T @ grad_h1
            grad_b1 = grad_h1.squeeze()

            # SGD update
            self.W3 -= learning_rate * grad_W3
            self.b3 -= learning_rate * grad_b3
            self.W2 -= learning_rate * grad_W2
            self.b2 -= learning_rate * grad_b2
            self.W1 -= learning_rate * grad_W1
            self.b1 -= learning_rate * grad_b1

        return total_loss / max(batch_size, 1)


class NeuralReconfigAgent:
    """Neural network-based reconfiguration agent.

    Uses a small MLP Q-network with experience replay and a target
    network for stable online learning. Designed to run on CPU within
    the 1-15ms loop budget of the adaptive firmware layer.

    Key features:
    - Online learning (no offline pre-training required)
    - Experience replay buffer (breaks temporal correlation)
    - Target network (periodic copy for stable Q-learning)
    - Epsilon-greedy exploration with decay
    - Informed initialization via workload-class priors
    - Concept drift detection (same ADWIN-based detector)
    """

    # Workload-class priors (same as tabular agent for consistency)
    _PRIORS: dict[str, list[tuple[int, float]]] = {
        "compute_bound": [(0, 0.7), (2, 0.5), (3, 0.3), (1, 0.2)],
        "memory_bound":  [(1, 0.7), (2, 0.5), (3, 0.3), (0, 0.2)],
        "balanced":      [(2, 0.6), (0, 0.5), (1, 0.5), (3, 0.4)],
        "unknown":       [(2, 0.4), (0, 0.3), (1, 0.3), (3, 0.3)],
    }

    def __init__(
        self,
        configs: list[AcceleratorConfig],
        learning_rate: float = 0.001,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        energy_weight: float = 0.15,
        buffer_capacity: int = 1000,
        batch_size: int = 32,
        target_update_freq: int = 50,
        train_freq: int = 4,
        hidden_dim: int = 64,
        drift_window: int = 30,
        drift_threshold: float = 2.0,
    ) -> None:
        """Initialize the neural network agent.

        Args:
            configs: Available accelerator configurations.
            learning_rate: SGD learning rate.
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Per-step decay factor for epsilon.
            energy_weight: Weight for energy in reward.
            buffer_capacity: Replay buffer size.
            batch_size: Mini-batch size for training.
            target_update_freq: Steps between target network updates.
            train_freq: Steps between training updates.
            hidden_dim: MLP hidden layer size.
            drift_window: Window size for drift detection.
            drift_threshold: Threshold for drift detection.
        """
        self.configs = configs
        self.n_actions = len(configs)
        self.config_by_id = {c.config_id: c for c in configs}
        self.lr = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.energy_weight = energy_weight
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq

        # Networks
        self.policy_net = NeuralPolicy(state_dim=8, n_actions=self.n_actions, hidden_dim=hidden_dim)
        self.target_net = NeuralPolicy(state_dim=8, n_actions=self.n_actions, hidden_dim=hidden_dim)
        self.target_net.copy_from(self.policy_net)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Drift detector
        self.drift_detector = DriftDetector(
            window_size=drift_window,
            threshold=drift_threshold,
        )

        # State
        self.stats = NNAgentStats(exploration_rate=self.epsilon)
        self._last_state_vec: np.ndarray | None = None
        self._last_action: int = 0
        self._steps_since_target_update = 0
        self._steps_since_train = 0

    def select_action(self, telemetry: TelemetryVector) -> int:
        """Select a config using epsilon-greedy over the policy network's Q-values."""
        state_vec = np.array(telemetry.to_feature_vector(), dtype=np.float32)
        self._last_state_vec = state_vec

        if np.random.random() < self.epsilon:
            action = int(np.random.randint(self.n_actions))
        else:
            q_values = self.policy_net.forward(state_vec)
            # Break ties randomly
            max_q = q_values.max()
            best_actions = np.where(q_values == max_q)[0]
            action = int(np.random.choice(best_actions))

        self._last_action = action
        return action

    def update(
        self,
        telemetry: TelemetryVector,
        action: int,
        reward: float,
    ) -> None:
        """Update the policy network based on observed reward.

        Stores the experience in the replay buffer and periodically
        trains on a mini-batch sampled from it.
        """
        state_vec = np.array(telemetry.to_feature_vector(), dtype=np.float32)

        # Store transition
        if self._last_state_vec is not None:
            self.replay_buffer.push(self._last_state_vec, action, reward, state_vec)

        # Train periodically
        self._steps_since_train += 1
        if self._steps_since_train >= self.train_freq and len(self.replay_buffer) >= self.batch_size:
            self._train_step()
            self._steps_since_train = 0

        # Update target network periodically
        self._steps_since_target_update += 1
        if self._steps_since_target_update >= self.target_update_freq:
            self.target_net.copy_from(self.policy_net)
            self._steps_since_target_update = 0

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Drift detection
        drift = self.drift_detector.update(reward)
        if drift:
            self.epsilon = min(0.3, self.epsilon * 3.0)

        # Stats
        self.stats.total_steps += 1
        self.stats.total_reward += reward
        self.stats.avg_reward = self.stats.total_reward / self.stats.total_steps
        self.stats.exploration_rate = self.epsilon
        self.stats.config_selections[action] = self.stats.config_selections.get(action, 0) + 1

    def _train_step(self) -> None:
        """Train on a mini-batch from the replay buffer."""
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        # Compute targets using target network: Q_target(s', a*) for best a*
        next_q = self.target_net.forward(next_states)  # (batch, n_actions)
        best_next_actions = np.argmax(next_q, axis=1)
        next_q_values = next_q[np.arange(len(rewards)), best_next_actions]
        targets = rewards + 0.99 * next_q_values  # gamma=0.99

        # Train
        loss = self.policy_net.train_step(states, actions, targets, learning_rate=self.lr)
        self.stats.avg_loss = 0.9 * self.stats.avg_loss + 0.1 * loss

    def compute_reward(
        self,
        exec_result,  # ExecutionResult
        energy_budget_remaining: float = 1.0,
    ) -> float:
        """Compute reward (same formula as tabular agent for fair comparison)."""
        max_throughput = max(c.compute_throughput for c in self.configs)
        throughput_score = exec_result.throughput_gops / max_throughput

        config = self.config_by_id[exec_result.config_id]
        min_energy = min(c.energy_per_op for c in self.configs)
        max_energy = max(c.energy_per_op for c in self.configs)
        if max_energy > min_energy:
            energy_score = 1.0 - (config.energy_per_op - min_energy) / (max_energy - min_energy)
        else:
            energy_score = 0.5

        cache_bonus = 0.2 if exec_result.cache_hit else 0.0
        reconfig_penalty = min(0.25, exec_result.reconfig_time_ms / 30.0)
        budget_penalty = 0.0 if energy_budget_remaining > 0.2 else 0.2

        reward = (
            (1.0 - self.energy_weight) * throughput_score
            + self.energy_weight * energy_score
            + cache_bonus
            - reconfig_penalty
            - budget_penalty
        )

        return max(0.0, min(1.0, reward))

    def get_policy(self, telemetry: TelemetryVector) -> int:
        """Get the greedy policy for a given telemetry."""
        state_vec = np.array(telemetry.to_feature_vector(), dtype=np.float32)
        q_values = self.policy_net.forward(state_vec)
        return int(np.argmax(q_values))

    def get_policy_dict(self) -> dict[str, int]:
        """Get the greedy policy for all known workload classes.

        Returns a dict mapping workload_class -> best config_id.
        Compatible with the tabular agent's get_policy() interface.
        """
        from ..observation.telemetry import TelemetryVector
        policy: dict[str, int] = {}
        for wc in ["compute_bound", "memory_bound", "balanced", "unknown"]:
            t = TelemetryVector(
                op_type="Conv2d", flops=1e9, memory_bytes=1e7,
                arithmetic_intensity=100.0, workload_class=wc,
                current_config_id=None, cache_loaded_configs=[],
                energy_budget_remaining=1.0, latency_target_ms=50.0,
            )
            policy[wc] = self.get_policy(t)
        return policy

    def reset(self) -> None:
        """Reset agent state for a new episode (keeps learned weights)."""
        self.epsilon = 0.3
        self.drift_detector.reset()
        self.stats = NNAgentStats(exploration_rate=self.epsilon)
        self._last_state_vec = None
        self._last_action = 0
        self._steps_since_target_update = 0
        self._steps_since_train = 0
