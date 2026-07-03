"""Look-ahead scheduling for the adaptive firmware layer.

The current per-op decision is suboptimal: the agent decides which config
to use for each op in isolation, not knowing what comes next. In reality,
the model graph is known ahead of inference, so the agent can look ahead
and prefetch configs to overlap reconfiguration with execution.

This module adds:
1. A look-ahead window: the agent sees the next N traces
2. A "prefetch" action: the agent can pre-load a config for future use
3. Overlapping reconfiguration: prefetch happens during current op's execution
4. A revised reward that accounts for the prefetch benefit

The key insight: reconfiguration is expensive (3-8ms per bitstream), but
if you can start loading a config while the current op is still running,
the effective reconfig cost is much lower. Look-ahead scheduling exploits
this to amortize reconfiguration across multiple ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Iterator

import numpy as np

from ..observation.telemetry import TelemetryVector, WorkloadTrace
from ..hardware.simulator import HardwareSimulator, ExecutionResult
from ..hardware.configs import AcceleratorConfig
from ..agent.rl_agent import ReconfigAgent
from ..agent.drift_detector import DriftDetector


@dataclass
class LookaheadConfig:
    """A planned reconfiguration based on look-ahead.

    If the agent decides config B is needed for the next op, it can
    start loading B during the current op, so B is ready when needed.
    """

    config_id: int
    estimated_load_time_ms: float
    trigger_after_step: int  # Step at which this config will be needed
    current_progress_ms: float = 0.0  # How much of the load is done


class LookaheadAgent:
    """Reconfiguration agent with look-ahead scheduling.

    Extends the basic agent by:
    1. Maintaining a "planned queue" of future config switches
    2. Overlapping reconfiguration with execution
    3. Using the look-ahead window to anticipate workload changes

    The agent looks at the next N workload classes and decides:
    - Stay on the current config (if it's still best)
    - Pre-load a different config (if the workload class is about to change)
    - Pre-load multiple configs (if the workload will cycle)
    """

    def __init__(
        self,
        configs: list[AcceleratorConfig],
        lookahead_window: int = 5,
        overlap_factor: float = 0.7,
        learning_rate: float = 0.15,
        epsilon_start: float = 0.2,
        energy_weight: float = 0.15,
    ) -> None:
        """Initialize the look-ahead agent.

        Args:
            configs: Available accelerator configurations.
            lookahead_window: How many future ops to look ahead.
            overlap_factor: Fraction of reconfig time that can be
                          overlapped with current execution (0-1).
                          0.7 = 70% of the reconfig happens during
                          the current op, 30% is still a penalty.
            learning_rate: Q-value update rate.
            epsilon_start: Initial exploration rate.
            energy_weight: Weight for energy in reward.
        """
        self.configs = configs
        self.config_by_id = {c.config_id: c for c in configs}
        self.lookahead_window = lookahead_window
        self.overlap_factor = overlap_factor
        self.energy_weight = energy_weight

        # Q-table (same as basic agent, but with look-ahead state)
        self.lookahead_states = ["single", "stable", "changing", "cycling"]
        self.q_table: dict[str, np.ndarray] = {
            state: np.zeros(len(configs)) for state in self.lookahead_states
        }

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

        # Planned reconfig queue
        self.planned: deque[LookaheadConfig] = deque()

        # Stats
        self.steps = 0
        self.total_reward = 0.0
        self.overlapped_reconfigs = 0
        self.total_reconfigs = 0

        # Informed initialization
        self._init_q_values()

    def _init_q_values(self) -> None:
        """Seed Q-values with look-ahead-aware priors.

        The agent learns that:
        - "single" (one workload class coming) -> use the best config for that class
        - "stable" (same class for next N ops) -> use that class's best config, no switching
        - "changing" (different class coming) -> pre-load the new class's config
        - "cycling" (multiple classes alternating) -> use the most common class's config
        """
        priors: dict[str, list[tuple[int, float]]] = {
            "single":   [(2, 0.6), (0, 0.5), (1, 0.5), (3, 0.4)],  # balanced default
            "stable":   [(2, 0.7), (0, 0.5), (1, 0.5), (3, 0.4)],  # stay on best
            "changing": [(2, 0.5), (0, 0.5), (1, 0.5), (3, 0.4)],  # anticipate change
            "cycling":  [(2, 0.5), (0, 0.4), (1, 0.4), (3, 0.3)],  # pick most common
        }
        for state, rankings in priors.items():
            for cid, score in rankings:
                if cid < len(self.configs):
                    self.q_table[state][cid] = score

    def _classify_lookahead(self, future_traces: list[WorkloadTrace]) -> str:
        """Classify the look-ahead pattern.

        - "single": only one workload class coming
        - "stable": all same class
        - "changing": workload class changes once
        - "cycling": workload class alternates
        """
        if not future_traces:
            return "single"

        classes = [t.workload_class for t in future_traces]
        unique = set(classes)

        if len(unique) == 1:
            return "stable"
        if len(unique) == 2 and classes[0] != classes[1]:
            # Check if it changes and stays
            mid = len(classes) // 2
            if len(set(classes[:mid])) == 1 and len(set(classes[mid:])) == 1 and classes[0] != classes[mid]:
                return "changing"
        if len(unique) >= 3:
            return "cycling"
        return "single"

    def select_action(
        self,
        current_telemetry: TelemetryVector,
        future_traces: list[WorkloadTrace],
        current_config_id: int,
    ) -> int:
        """Select a config considering the look-ahead window.

        Args:
            current_telemetry: Telemetry for the current op.
            future_traces: Next N traces (look-ahead window).
            current_config_id: Currently loaded config.

        Returns:
            config_id to load (may be the same as current, meaning stay).
        """
        # Classify the look-ahead pattern
        lookahead_state = self._classify_lookahead(future_traces)
        current_class = current_telemetry.workload_class

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            return int(np.random.randint(len(self.configs)))

        # Greedy action based on Q-value
        q_values = self.q_table[lookahead_state]
        max_q = q_values.max()
        best_actions = np.where(q_values == max_q)[0]
        action = int(np.random.choice(best_actions))

        # Heuristic override: if the look-ahead is "stable" and the current
        # config is already loaded, prefer to stay (don't reconfigure)
        if lookahead_state == "stable" and action != current_config_id:
            # Check if current config is the best for the current class
            current_class_best = self._best_config_for_class(current_class)
            if current_class_best == current_config_id:
                return current_config_id

        return action

    def _best_config_for_class(self, workload_class: str) -> int:
        """Return the best config for a workload class (from priors)."""
        priors = {
            "compute_bound": 0,
            "memory_bound": 1,
            "balanced": 2,
            "unknown": 2,
        }
        return priors.get(workload_class, 2)

    def update(
        self,
        lookahead_state: str,
        action: int,
        reward: float,
    ) -> None:
        """Update Q-values based on observed reward."""
        self.q_table[lookahead_state][action] += 0.15 * (reward - self.q_table[lookahead_state][action])
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1
        self.total_reward += reward

    def record_reconfig(self, overlapped: bool) -> None:
        """Record whether a reconfiguration was overlapped with execution."""
        self.total_reconfigs += 1
        if overlapped:
            self.overlapped_reconfigs += 1

    def compute_reward_for_result(
        self,
        exec_result,  # ExecutionResult
        energy_budget_remaining: float = 1.0,
    ) -> float:
        """Compute the reward for an execution result.

        Same formula as the tabular agent for fair comparison:
        - Throughput component (normalized by max possible)
        - Energy component (lower is better)
        - Cache hit bonus (staying is good)
        - Reconfig penalty (already reduced by overlap in the caller)
        - Budget penalty
        """
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

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.steps, 1)

    @property
    def overlap_rate(self) -> float:
        return self.overlapped_reconfigs / max(self.total_reconfigs, 1)

    def get_policy(self) -> dict[str, int]:
        return {
            state: int(np.argmax(q)) for state, q in self.q_table.items()
        }

    def reset(self) -> None:
        self.epsilon = 0.2
        self.planned.clear()
        self.steps = 0
        self.total_reward = 0.0
        self.overlapped_reconfigs = 0
        self.total_reconfigs = 0


def compute_overlapped_reconfig_cost(
    reconfig_time_ms: float,
    current_op_time_ms: float,
    overlap_factor: float = 0.7,
) -> tuple[float, float]:
    """Compute the effective reconfig cost when overlapped with current op.

    If the reconfiguration can be hidden behind the current op's execution,
    only the non-overlapped portion counts as a penalty.

    Returns:
        (effective_cost_ms, non_overlapped_ms)
    """
    overlapped = min(reconfig_time_ms * overlap_factor, current_op_time_ms)
    non_overlapped = max(0.0, reconfig_time_ms - overlapped)
    return non_overlapped, reconfig_time_ms - non_overlapped
