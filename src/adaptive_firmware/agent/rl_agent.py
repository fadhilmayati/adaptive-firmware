"""RL agent for runtime hardware reconfiguration.

Uses a contextual bandit approach — the simplest online-learning RL
formulation that fits this problem. The agent observes a state vector
(workload + hardware telemetry), selects a config (action), and receives
a reward (throughput / energy efficiency). It learns online, adapting
its policy as workloads change.

Why contextual bandit instead of full RL (PPO/SAC)?
- No sequential dependency between decisions (each op is independent)
- Immediate reward (we know the execution result right away)
- Simpler, more stable online learning
- No need for replay buffers or target networks

The agent uses a softmax policy over learned Q-values, with an
epsilon-greedy exploration strategy that decays over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .drift_detector import DriftDetector
from ..observation.telemetry import TelemetryVector
from ..hardware.configs import AcceleratorConfig


@dataclass
class AgentStats:
    """Running statistics for the agent."""

    total_steps: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    config_selections: dict[int, int] = field(default_factory=dict)
    best_config_found: int | None = None
    best_avg_reward: float = float("-inf")
    exploration_rate: float = 0.0


class ReconfigAgent:
    """Contextual bandit agent for hardware reconfiguration.

    Maintains a Q-table indexed by a discretized workload class, with
    one entry per action (accelerator config). Updates Q-values online
    using a learning rate, with epsilon-greedy exploration.

    The state is the workload class ("compute_bound", "memory_bound",
    "balanced") — a natural discretization that maps directly to which
    config is optimal. This is intentionally simple for the PoC; a
    production version would use a neural network policy.
    """

    STATES = ["compute_bound", "memory_bound", "balanced", "unknown"]
    DEFAULT_ENERGY_WEIGHT = 0.15

    def __init__(
        self,
        configs: list[AcceleratorConfig],
        learning_rate: float = 0.1,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        energy_weight: float | None = None,
        drift_window: int = 30,
        drift_threshold: float = 2.0,
    ) -> None:
        """Initialize the agent.

        Args:
            configs: Available accelerator configurations (actions).
            learning_rate: Q-value update rate (alpha).
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Per-step decay factor for epsilon.
            energy_weight: Weight for energy in reward (0 = throughput only,
                          1 = energy only). Defaults to DEFAULT_ENERGY_WEIGHT.
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
        self.energy_weight = (
            energy_weight if energy_weight is not None else self.DEFAULT_ENERGY_WEIGHT
        )

        # Q-table: state -> action -> Q-value
        self.q_table: dict[str, np.ndarray] = {
            state: np.zeros(self.n_actions) for state in self.STATES
        }

        # Informed initialization: seed Q-values based on what we know
        # about which configs are best for each workload class. This
        # eliminates the cold-start exploration phase for the obvious
        # mappings, letting the agent focus exploration on nuanced cases.
        self._init_q_values()

        self.drift_detector = DriftDetector(
            window_size=drift_window,
            threshold=drift_threshold,
        )
        self.stats = AgentStats(
            exploration_rate=self.epsilon,
        )
        self._last_state: str = "unknown"
        self._last_action: int = 0

    def _init_q_values(self) -> None:
        """Seed Q-values with workload-class → config priors.

        These are approximate but correct orderings, eliminating the
        cold-start cost of discovering obvious mappings like
        "memory_bound → HIGH_BANDWIDTH."
        """
        # Prior knowledge: which config is best for each workload class
        # (config_id, confidence_score)
        priors: dict[str, list[tuple[int, float]]] = {
            "compute_bound": [(0, 0.7), (2, 0.5), (3, 0.3), (1, 0.2)],
            "memory_bound":  [(1, 0.7), (2, 0.5), (3, 0.3), (0, 0.2)],
            "balanced":      [(2, 0.6), (0, 0.5), (1, 0.5), (3, 0.4)],
            "unknown":       [(2, 0.4), (0, 0.3), (1, 0.3), (3, 0.3)],
        }
        for state, rankings in priors.items():
            for config_id, score in rankings:
                if config_id < self.n_actions:
                    self.q_table[state][config_id] = score

    def _discretize_state(self, telemetry: TelemetryVector) -> str:
        """Map telemetry to a discrete state for the Q-table."""
        return telemetry.workload_class if telemetry.workload_class in self.q_table else "unknown"

    def select_action(self, telemetry: TelemetryVector) -> int:
        """Select a config (action) given the current telemetry (state).

        Uses epsilon-greedy: with probability epsilon, explore randomly;
        otherwise exploit the best-known config for this state.
        """
        state = self._discretize_state(telemetry)
        self._last_state = state

        if np.random.random() < self.epsilon:
            # Explore
            action = np.random.randint(self.n_actions)
        else:
            # Exploit: pick the action with highest Q-value
            q_values = self.q_table[state]
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
        """Update Q-values based on observed reward.

        Uses the standard Q-learning update:
            Q(s, a) <- Q(s, a) + alpha * (reward - Q(s, a))

        Also decays epsilon, checks for drift, and updates stats.
        """
        state = self._discretize_state(telemetry)

        # Q-value update
        self.q_table[state][action] += self.lr * (reward - self.q_table[state][action])

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Drift detection
        drift = self.drift_detector.update(reward)
        if drift:
            # Boost exploration after drift
            self.epsilon = min(0.3, self.epsilon * 3.0)

        # Update stats
        self.stats.total_steps += 1
        self.stats.total_reward += reward
        self.stats.avg_reward = self.stats.total_reward / self.stats.total_steps
        self.stats.exploration_rate = self.epsilon
        self.stats.config_selections[action] = (
            self.stats.config_selections.get(action, 0) + 1
        )

        # Track best config per state
        current_best = self.q_table[state].max()
        if current_best > self.stats.best_avg_reward:
            self.stats.best_avg_reward = current_best
            self.stats.best_config_found = int(np.argmax(self.q_table[state]))

    def compute_reward(
        self,
        exec_result,  # ExecutionResult from HardwareSimulator
        energy_budget_remaining: float = 1.0,
    ) -> float:
        """Compute the reward for an execution result.

        Reward balances throughput and energy efficiency, scaled by
        whether the config is a cache hit (reconfiguration is expensive).
        A cache hit gives a bonus because it avoided reconfig overhead.

        The reward is normalized to roughly [0, 1] range.
        """
        # Throughput component (normalize by max possible throughput)
        max_throughput = max(c.compute_throughput for c in self.configs)
        throughput_score = exec_result.throughput_gops / max_throughput

        # Energy component (lower energy = higher score)
        # Normalize: energy_per_op inversely related to efficiency
        config = self.config_by_id[exec_result.config_id]
        min_energy = min(c.energy_per_op for c in self.configs)
        max_energy = max(c.energy_per_op for c in self.configs)
        if max_energy > min_energy:
            energy_score = 1.0 - (config.energy_per_op - min_energy) / (max_energy - min_energy)
        else:
            energy_score = 0.5

        # Cache hit bonus: staying on a working config is strongly preferred.
        # Real workloads have temporal stability — if the last config worked
        # for this workload class, switching is almost always wasted.
        cache_bonus = 0.2 if exec_result.cache_hit else 0.0

        # Reconfig penalty: scaled up. Real reconfiguration is expensive
        # (energy + latency + wear) and should not be undertaken lightly.
        reconfig_penalty = min(0.25, exec_result.reconfig_time_ms / 30.0)

        # Energy budget penalty
        budget_penalty = 0.0 if energy_budget_remaining > 0.2 else 0.2

        reward = (
            (1.0 - self.energy_weight) * throughput_score
            + self.energy_weight * energy_score
            + cache_bonus
            - reconfig_penalty
            - budget_penalty
        )

        return max(0.0, min(1.0, reward))

    def get_policy(self) -> dict[str, int]:
        """Return the current greedy policy (state -> best action)."""
        return {
            state: int(np.argmax(q)) for state, q in self.q_table.items()
        }

    def reset(self) -> None:
        """Reset agent state for a new episode (keeps learned Q-values)."""
        self.epsilon = 0.3  # Reset exploration
        self.drift_detector.reset()
        self.stats = AgentStats(exploration_rate=self.epsilon)
