"""Thompson sampling bandit agent for runtime hardware reconfiguration.

Uses Thompson sampling, a Bayesian bandit algorithm. Maintains a Beta
posterior for each (state, action) pair, samples from it, and picks the
action with the highest sample.

This provides a principled algorithmic contrast to epsilon-greedy Q-learning
(tabular agent). Thompson sampling explores proportionally to posterior
uncertainty, while Q-learning explores uniformly at random.

Key difference visible in benchmarks:
  - Large heterogeneous workloads: Thompson discovers the optimal config
    faster because it explores uncertain arms more often.
  - Small homogeneous workloads: Thompson explores too much because the
    posterior stays wide for seldom-seen states.

Cache-aware mode (cache_aware=True):
  Debiases the reward signal to remove the cache bonus/reconfiguration penalty
  from posterior updates, then re-adds the cache effect at action-selection
  time. This prevents the agent from conflating "this config is good for the
  workload" with "this config was lucky enough to have a warm cache."
"""

from __future__ import annotations

import numpy as np

from .drift_detector import DriftDetector
from .rl_agent import AgentStats
from ..observation.telemetry import TelemetryVector
from ..hardware.configs import AcceleratorConfig


class UCBAgent:
    """Thompson sampling bandit agent for hardware reconfiguration.

    For each workload class (state), maintains independent Beta posteriors
    for each accelerator config (action). Action selection samples from the
    posterior and picks the max — naturally balancing exploration of
    uncertain arms with exploitation of configs known to be good.

    When cache_aware=True, the agent debiases the reward signal to remove
    the cache bonus/reconfiguration penalty from posterior updates, then
    re-adds these effects at action-selection time. This lets the agent
    learn each config's true steady-state quality without conflating it
    with transient cache state.
    """

    STATES = ["compute_bound", "memory_bound", "balanced", "unknown"]
    DEFAULT_ENERGY_WEIGHT = 0.15
    PRIOR_STRENGTH = 4
    CACHE_BONUS = 0.2
    _AMORTIZATION_HORIZON = 3  # Empirically optimal: balances switch-cost avoidance with exploration

    def __init__(
        self,
        configs: list[AcceleratorConfig],
        energy_weight: float | None = None,
        drift_window: int = 30,
        drift_threshold: float = 2.0,
        cache_aware: bool = False,
    ) -> None:
        self.configs = configs
        self.n_actions = len(configs)
        self.config_by_id = {c.config_id: c for c in configs}
        self.energy_weight = (
            energy_weight if energy_weight is not None else self.DEFAULT_ENERGY_WEIGHT
        )
        self.cache_aware = cache_aware
        self._last_cache_effect: float = 0.0

        self.alpha: dict[str, np.ndarray] = {}
        self.beta: dict[str, np.ndarray] = {}
        self._init_posteriors()

        self.drift_detector = DriftDetector(
            window_size=drift_window,
            threshold=drift_threshold,
        )
        self.stats = AgentStats()
        self.epsilon = 0.0
        self._last_state: str = "unknown"
        self._last_action: int = 0

    def _init_posteriors(self) -> None:
        """Seed Beta posteriors with prior knowledge of workload->config mappings."""
        priors: dict[str, list[tuple[int, float]]] = {
            "compute_bound": [(0, 0.7), (2, 0.5), (3, 0.3), (1, 0.2)],
            "memory_bound":  [(1, 0.7), (2, 0.5), (3, 0.3), (0, 0.2)],
            "balanced":      [(2, 0.6), (0, 0.5), (1, 0.5), (3, 0.4)],
            "unknown":       [(2, 0.4), (0, 0.3), (1, 0.3), (3, 0.3)],
        }
        s = self.PRIOR_STRENGTH
        for state in self.STATES:
            alphas = np.ones(self.n_actions)
            betas = np.ones(self.n_actions)
            if state in priors:
                for config_id, mean in priors[state]:
                    if config_id < self.n_actions:
                        alphas[config_id] = 1.0 + s * mean
                        betas[config_id] = 1.0 + s * (1.0 - mean)
            self.alpha[state] = alphas
            self.beta[state] = betas

    def _discretize_state(self, telemetry: TelemetryVector) -> str:
        """Map telemetry to a discrete state string."""
        return (
            telemetry.workload_class
            if telemetry.workload_class in self.alpha
            else "unknown"
        )

    def select_action(self, telemetry: TelemetryVector) -> int:
        """Select config via Thompson sampling.

        Samples from each config's Beta posterior and picks the config
        with the highest sampled value.

        In cache_aware mode, adjusts cold config samples down by an
        amortized switching cost (one-time penalty + lost cache bonus
        divided over a reasonable horizon). This prevents the agent from
        conflating "this config is better" with "I'm already cached on
        this config," while still accounting for the real (but amortized)
        cost of switching.
        """
        state = self._discretize_state(telemetry)
        self._last_state = state

        samples = np.random.beta(self.alpha[state], self.beta[state])

        if self.cache_aware:
            cached = telemetry.cache_loaded_configs
            for j in range(self.n_actions):
                if j not in cached:
                    cfg = self.config_by_id[j]
                    penalty = min(0.25, cfg.reconfig_time_ms / 30.0)
                    # Amortize the one-time switch cost over H future traces
                    amortized = (self.CACHE_BONUS + penalty) / self._AMORTIZATION_HORIZON
                    samples[j] -= amortized

        max_val = samples.max()
        best_actions = np.where(samples == max_val)[0]
        action = int(np.random.choice(best_actions))

        self._last_action = action
        return action

    def update(
        self,
        telemetry: TelemetryVector,
        action: int,
        reward: float,
    ) -> None:
        """Update Beta posterior based on observed reward.

        In cache_aware mode, debiases the reward by removing the cache
        effect (cache bonus + reconfiguration penalty) before updating
        the posterior. This ensures the posterior reflects the config's
        true steady-state quality, not a transient cache windfall or
        cold-start penalty.
        """
        state = self._discretize_state(telemetry)

        if self.cache_aware:
            debiased = reward - self._last_cache_effect
            debiased = max(0.0, min(1.0, debiased))
        else:
            debiased = reward

        self.alpha[state][action] += debiased
        self.beta[state][action] += 1.0 - debiased

        drift = self.drift_detector.update(reward)
        if drift:
            for state in self.STATES:
                self.alpha[state] = np.maximum(self.alpha[state] * 0.5, 1.0)
                self.beta[state] = np.maximum(self.beta[state] * 0.5, 1.0)

        self.stats.total_steps += 1
        self.stats.total_reward += reward
        self.stats.avg_reward = self.stats.total_reward / self.stats.total_steps
        self.stats.config_selections[action] = (
            self.stats.config_selections.get(action, 0) + 1
        )

        posterior_mean = self.alpha[state] / (
            self.alpha[state] + self.beta[state] + 1e-10
        )
        current_best = np.max(posterior_mean)
        if current_best > self.stats.best_avg_reward:
            self.stats.best_avg_reward = current_best
            self.stats.best_config_found = int(np.argmax(posterior_mean))

    def compute_reward(
        self,
        exec_result,
        energy_budget_remaining: float = 1.0,
    ) -> float:
        """Compute reward for an execution result, normalized to [0, 1]."""
        max_throughput = max(c.compute_throughput for c in self.configs)
        throughput_score = exec_result.throughput_gops / max_throughput

        config = self.config_by_id[exec_result.config_id]
        min_energy = min(c.energy_per_op for c in self.configs)
        max_energy = max(c.energy_per_op for c in self.configs)
        if max_energy > min_energy:
            energy_score = 1.0 - (
                config.energy_per_op - min_energy
            ) / (max_energy - min_energy)
        else:
            energy_score = 0.5

        cache_bonus = self.CACHE_BONUS if exec_result.cache_hit else 0.0
        reconfig_penalty = min(0.25, exec_result.reconfig_time_ms / 30.0)
        budget_penalty = 0.0 if energy_budget_remaining > 0.2 else 0.2

        reward = (
            (1.0 - self.energy_weight) * throughput_score
            + self.energy_weight * energy_score
            + cache_bonus
            - reconfig_penalty
            - budget_penalty
        )

        if self.cache_aware:
            self._last_cache_effect = cache_bonus - reconfig_penalty

        return max(0.0, min(1.0, reward))

    def get_policy(self) -> dict[str, int]:
        """Return the current greedy policy by posterior mean."""
        return {
            state: int(np.argmax(
                self.alpha[state] / (self.alpha[state] + self.beta[state] + 1e-10)
            ))
            for state in self.STATES
        }

    def reset(self) -> None:
        """Reset agent state for a new episode (keeps learned posteriors)."""
        self.drift_detector.reset()
        self.stats = AgentStats()
