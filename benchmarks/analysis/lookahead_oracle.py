"""Look-ahead oracle: optimal dynamic switching with reconfiguration cost.

The greedy oracle switches on every workload class change — which is naive
because it ignores reconfiguration cost. On frequent-switch workloads it
underperforms even a simple static config.

The look-ahead oracle knows the entire future trace sequence and computes
the globally optimal switching policy via dynamic programming.

DP state: (loaded_config slot1, cached_config slot2)
- slot1 = most recently used config (or -1 if empty)
- slot2 = second most recently used config (or -1 if empty)
- With a 2-slot LRU cache, switching between two cached configs costs 0.
  This is critical — the naive DP that charges reconfig on every switch
  is too conservative and underperforms the greedy oracle.
"""

from __future__ import annotations

import numpy as np

from adaptive_firmware.hardware.configs import AcceleratorConfig, CONFIG_PRESETS
from adaptive_firmware.observation.telemetry import WorkloadTrace
from adaptive_firmware.runtime.middleware import AdaptiveMiddleware, StepLog, EpisodeReport


OPTIMAL_MAP = {
    "compute_bound": 0,  # HIGH_COMPUTE
    "memory_bound":  1,  # HIGH_BANDWIDTH
    "balanced":      2,  # BALANCED
    "low_power":     3,  # LOW_POWER
}

EMPTY = -1  # Sentinel for empty cache slot
N_CONFIGS = 4  # Number of hardware configs


def _cache_state_encode(slot1: int, slot2: int) -> int:
    """Encode (slot1, slot2) into a single integer 0..24."""
    s1 = slot1 if slot1 >= 0 else N_CONFIGS  # -1 → 4
    s2 = slot2 if slot2 >= 0 else N_CONFIGS
    return s1 * (N_CONFIGS + 1) + s2


def _cache_state_decode(state: int) -> tuple[int, int]:
    s1 = state // (N_CONFIGS + 1)
    s2 = state % (N_CONFIGS + 1)
    if s1 >= N_CONFIGS:
        s1 = EMPTY
    if s2 >= N_CONFIGS:
        s2 = EMPTY
    return s1, s2


N_CACHE_STATES = (N_CONFIGS + 1) * (N_CONFIGS + 1)


def _apply_action_to_cache(slot1: int, slot2: int, action: int) -> tuple[int, int]:
    """Update the LRU cache state after taking an action."""
    if action == slot1:
        return (slot1, slot2)
    elif action == slot2:
        return (action, slot1)
    else:
        # Cache miss — evict LRU (slot2), promote slot1, load action
        return (action, slot1)


class LookAheadOracleMiddleware(AdaptiveMiddleware):
    """Middleware with perfect look-ahead — the TRUE upper bound.

    Uses dynamic programming over the full trace sequence, modeling the
    2-slot LRU cache explicitly. This is necessary because switching
    between two cached configs costs 0 (warm cache), so charging reconfig
    on every switch (as the old DP did) makes the oracle too conservative.
    """

    def __init__(
        self,
        configs: list[AcceleratorConfig],
        energy_weight: float = 0.15,
    ) -> None:
        super().__init__(configs=configs, energy_weight=energy_weight)
        self.energy_weight = energy_weight

    def run_episode(
        self,
        traces: list[WorkloadTrace],
        tenant_id: str = "default",
    ) -> EpisodeReport:
        n = len(traces)
        K = len(self.configs)

        # Pre-compute base rewards (throughput + energy, no cache/reconfig)
        base = np.zeros((n, K), dtype=np.float64)
        for i, t in enumerate(traces):
            for k in range(K):
                base[i, k] = self._compute_base_reward(k, t)

        # Reconfig penalty for loading config k from scratch (cache miss)
        def reconf_penalty(k: int) -> float:
            cfg = self.configs[k]
            return min(0.25, cfg.reconfig_time_ms / 30.0)

        cache_bonus = 0.2

        # DP backward pass over (trace_index x cache_state)
        V = np.zeros((n + 1, N_CACHE_STATES), dtype=np.float64)
        # optimal_action[i][s] = best config for trace i when in cache state s
        optimal_action = np.zeros((n, N_CACHE_STATES), dtype=np.int32)

        for i in range(n - 1, -1, -1):
            for s in range(N_CACHE_STATES):
                slot1, slot2 = _cache_state_decode(s)

                best_val = -np.inf
                best_act = -1

                for j in range(K):
                    # Check if j is in cache
                    in_cache = (j == slot1) or (j == slot2)

                    if in_cache:
                        step_reward = base[i, j] + cache_bonus
                    else:
                        step_reward = base[i, j] - reconf_penalty(j)

                    step_reward = max(0.0, min(1.0, step_reward))
                    new_slot1, new_slot2 = _apply_action_to_cache(slot1, slot2, j)
                    new_s = _cache_state_encode(new_slot1, new_slot2)
                    total = step_reward + V[i + 1, new_s]

                    if total > best_val:
                        best_val = total
                        best_act = j

                V[i, s] = best_val
                optimal_action[i, s] = best_act

        # Forward pass: follow the optimal action table
        actions = np.zeros(n, dtype=np.int32)
        slot1, slot2 = EMPTY, EMPTY  # cache starts empty

        for i in range(n):
            s = _cache_state_encode(slot1, slot2)

            # At trace 0, no config is cached — every action is a miss.
            # At later traces, the optimal_action table handles this correctly
            # (it knows which configs are in cache from the state).
            best_act = optimal_action[i, s]

            # If the table somehow picked an invalid action, fall back
            if best_act < 0 or best_act >= K:
                # Pick the best config for this workload class
                best_act = OPTIMAL_MAP.get(traces[i].workload_class, 2)

            actions[i] = best_act
            slot1, slot2 = _apply_action_to_cache(slot1, slot2, best_act)

        # Execute the optimal policy in the simulator for reward computation
        self.reset()
        logs: list[StepLog] = []
        for i, trace in enumerate(traces):
            action = int(actions[i])
            result = self.simulator.execute(
                flops=trace.flops,
                memory_bytes=trace.memory_bytes,
                config_id=action,
            )
            config = self.simulator.configs[action]
            reward = self.agent.compute_reward(result, self.energy_budget)
            logs.append(StepLog(
                step=i, tenant_id=tenant_id,
                op_type=trace.op_type,
                workload_class=trace.workload_class,
                selected_config=action, config_name=config.name,
                exec_time_ms=result.exec_time_ms,
                reconfig_time_ms=result.reconfig_time_ms,
                total_time_ms=result.total_time_ms,
                energy_mj=result.energy_mj,
                reward=reward,
                cache_hit=result.cache_hit,
                drift_detected=False, epsilon=0.0,
            ))

        return self._build_report(logs)

    def _compute_base_reward(
        self, config_id: int, trace: WorkloadTrace,
    ) -> float:
        """Compute the throughput + energy component of the reward.

        No cache bonus or reconfiguration penalty — just the raw
        quality of matching this config to this workload trace.
        """
        config = self.configs[config_id]

        compute_time_s = trace.flops / (config.compute_throughput * 1e9)
        memory_time_s = trace.memory_bytes / (config.memory_bandwidth * 1e9)
        exec_time_s = max(compute_time_s, memory_time_s)

        max_throughput = max(c.compute_throughput for c in self.configs)
        throughput_gops = (
            (trace.flops / 1e9) / (exec_time_s if exec_time_s > 0 else 1e-9)
        )
        throughput_score = throughput_gops / max_throughput

        min_energy = min(c.energy_per_op for c in self.configs)
        max_energy = max(c.energy_per_op for c in self.configs)
        if max_energy > min_energy:
            energy_score = 1.0 - (
                (config.energy_per_op - min_energy) / (max_energy - min_energy)
            )
        else:
            energy_score = 0.5

        return (
            (1.0 - self.energy_weight) * throughput_score
            + self.energy_weight * energy_score
        )
