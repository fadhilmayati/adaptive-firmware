"""ProfileThenCommit agent: a production-realistic adaptive policy.

Strategy:
1. Profile phase: run each config for a short period, measure reward
2. Selection: pick the config with the best average reward
3. Commit phase: stick with the selected config, minimal exploration
4. Drift response: if workload distribution changes, re-profile

This is a realistic production policy because:
- Profiling cost is bounded and predictable
- The committed phase minimizes reconfiguration overhead
- Drift detection allows adaptation to genuine workload changes

When does it win?
- Workloads where one config is clearly best for the duration
- Long workloads where exploration cost dominates
- Production systems where reconfiguration is expensive

When does it lose?
- Workloads that switch optimal config mid-stream
- Very short workloads where profiling overhead is too much
- Workloads with subtle differences between configs
"""

from __future__ import annotations

import numpy as np

from .drift_detector import DriftDetector
from ..observation.telemetry import TelemetryVector
from ..hardware.configs import AcceleratorConfig


class ProfileThenCommitAgent:
    """A profile-then-commit adaptive agent.

    The agent profiles each config for `profile_steps` traces, then
    commits to the best-performing config for the rest of the workload.
    If concept drift is detected, it re-enters the profile phase.
    """

    def __init__(
        self,
        configs: list[AcceleratorConfig],
        profile_steps: int = 10,
        commit_epsilon: float = 0.02,
        drift_window: int = 30,
        drift_threshold: float = 2.0,
    ) -> None:
        """Initialize the agent.

        Args:
            configs: Available accelerator configurations.
            profile_steps: Number of traces to run per config during profiling.
            commit_epsilon: Exploration rate during commit phase (low).
            drift_window: Window size for drift detection.
            drift_threshold: Threshold for drift detection.
        """
        self.configs = configs
        self.config_by_id = {c.config_id: c for c in configs}
        self.n_actions = len(configs)
        self.profile_steps = profile_steps
        self.commit_epsilon = commit_epsilon

        # State
        self._phase: str = "profile"  # "profile" or "commit"
        self._profile_config_idx: int = 0
        self._profile_step_count: int = 0
        self._committed_config: int = 0
        self._profile_rewards: list[list[float]] = [[] for _ in configs]

        # Drift detection
        self.drift_detector = DriftDetector(
            window_size=drift_window,
            threshold=drift_threshold,
        )

        # Stats
        self.steps = 0
        self.total_reward = 0.0
        self.reprofiles = 0
        self.profile_complete = False

    def select_action(self, telemetry: TelemetryVector) -> int:
        """Select a config based on the current phase."""
        if self._phase == "profile":
            # During profile, round-robin through configs
            if self._profile_step_count >= self.profile_steps:
                # Move to next config
                self._profile_config_idx += 1
                self._profile_step_count = 0
                if self._profile_config_idx >= self.n_actions:
                    # Profile complete — select best and commit
                    self._commit()
                    return self._committed_config
            action = self._profile_config_idx
            self._profile_step_count += 1
            return action
        else:
            # Commit phase — minimal exploration
            if np.random.random() < self.commit_epsilon:
                return int(np.random.randint(self.n_actions))
            return self._committed_config

    def _commit(self) -> None:
        """End profile phase, select best config, enter commit phase."""
        avg_rewards = [
            np.mean(rewards) if rewards else float("-inf")
            for rewards in self._profile_rewards
        ]
        self._committed_config = int(np.argmax(avg_rewards))
        self._phase = "commit"
        self.profile_complete = True

    def update(self, telemetry: TelemetryVector, action: int, reward: float) -> None:
        """Update internal state based on observed reward."""
        self.steps += 1
        self.total_reward += reward

        if self._phase == "profile":
            # Record reward for the config being profiled
            if action < self.n_actions:
                self._profile_rewards[action].append(reward)
        else:
            # Commit phase — check for drift
            drift = self.drift_detector.update(reward)
            if drift:
                self.reprofiles += 1
                self._enter_reprofile()

    def _enter_reprofile(self) -> None:
        """Re-enter profile phase due to detected drift."""
        self._phase = "profile"
        self._profile_config_idx = 0
        self._profile_step_count = 0
        self._profile_rewards = [[] for _ in self.configs]
        self.profile_complete = False

    def compute_reward_for_result(
        self,
        exec_result,  # ExecutionResult
        energy_budget_remaining: float = 1.0,
    ) -> float:
        """Compute reward (same formula as other agents for fair comparison)."""
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
            (1.0 - 0.15) * throughput_score
            + 0.15 * energy_score
            + cache_bonus
            - reconfig_penalty
            - budget_penalty
        )
        return max(0.0, min(1.0, reward))

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.steps, 1)

    def get_policy_dict(self) -> dict[str, int]:
        """Get the committed config (or 0 if still profiling)."""
        return {"committed": self._committed_config}

    def reset(self) -> None:
        """Reset for a new episode."""
        self._phase = "profile"
        self._profile_config_idx = 0
        self._profile_step_count = 0
        self._committed_config = 0
        self._profile_rewards = [[] for _ in self.configs]
        self.drift_detector.reset()
        self.steps = 0
        self.total_reward = 0.0
        self.reprofiles = 0
        self.profile_complete = False
