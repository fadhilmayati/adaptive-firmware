"""Analysis engine: synthetic workloads, custom runners, metrics.

Provides the infrastructure for the four analyses:
1. Heterogeneity sweep — find the break-even threshold
2. Multi-seed confidence intervals — statistical validity
3. Reconfiguration cost sensitivity — practical bounds
4. Oracle gap — headroom quantification
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adaptive_firmware.hardware.configs import AcceleratorConfig, CONFIG_PRESETS
from adaptive_firmware.hardware.cgra_configs import CGRA_PRESETS, CGRA_CACHE_CAPACITY
from adaptive_firmware.observation.telemetry import WorkloadTrace
from adaptive_firmware.runtime.middleware import AdaptiveMiddleware
from adaptive_firmware.agent.rl_agent import ReconfigAgent
from adaptive_firmware.agent.neural_agent import NeuralReconfigAgent
from adaptive_firmware.agent.profile_agent import ProfileThenCommitAgent
from adaptive_firmware.agent.ucb_agent import UCBAgent
from benchmarks.analysis.lookahead_oracle import LookAheadOracleMiddleware


# ─── Synthetic workload generator ───────────────────────────────────────


def generate_synthetic_workload(
    seed: int = 42,
    n_traces: int = 1000,
    switch_prob: float = 0.0,
) -> list[WorkloadTrace]:
    """Generate a workload with controlled heterogeneity.

    The workload alternates between compute_bound and memory_bound traces.
    At each step, with probability `switch_prob`, the class flips.
    This gives a clean heterogeneity metric: expected switch fraction = switch_prob.

    Args:
        seed: Random seed for reproducibility.
        n_traces: Number of traces to generate.
        switch_prob: Probability of switching workload class each step.

    Returns:
        List of WorkloadTrace objects with controlled switch dynamics.
    """
    rng = random.Random(seed)

    # Two workload classes with clearly different optimal configs
    classes = ["compute_bound", "memory_bound"]

    # Class profiles: (typical_flops, typical_memory_bytes, op_type)
    profiles = {
        "compute_bound": (3e10, 1e9, "MatMul"),   # AI ≈ 30
        "memory_bound":  (2e9,  1e9, "Linear"),    # AI ≈ 2
    }

    traces: list[WorkloadTrace] = []
    current_class: str = rng.choice(classes)

    for _ in range(n_traces):
        # Maybe switch class
        if rng.random() < switch_prob:
            others = [c for c in classes if c != current_class]
            current_class = rng.choice(others)

        base_flops, base_mem, op_type = profiles[current_class]

        # Add noise (±20%) for realism while keeping classification clean
        flops = base_flops * rng.uniform(0.8, 1.2)
        mem = base_mem * rng.uniform(0.8, 1.2)

        traces.append(WorkloadTrace(
            op_type=op_type,
            flops=flops,
            memory_bytes=mem,
            batch_size=1,
            tensor_shapes=[],
            arithmetic_intensity=flops / max(mem, 1.0),
            workload_class=current_class,
        ))

    return traces


def compute_heterogeneity(traces: list[WorkloadTrace]) -> float:
    """Compute the observed heterogeneity of a trace list.

    Heterogeneity = fraction of consecutive trace pairs where
    the optimal config changes (according to the oracle mapping).

    Returns a value in [0, 1].
    """
    if len(traces) < 2:
        return 0.0

    optimal = {
        "compute_bound": 0,  # HIGH_COMPUTE
        "memory_bound":  1,  # HIGH_BANDWIDTH
        "balanced":      2,  # BALANCED
        "low_power":     3,  # LOW_POWER
    }

    switches = 0
    for i in range(1, len(traces)):
        prev_opt = optimal.get(traces[i - 1].workload_class, 2)
        curr_opt = optimal.get(traces[i].workload_class, 2)
        if prev_opt != curr_opt:
            switches += 1

    return switches / (len(traces) - 1)


# ─── Reconfiguration cost scaling ───────────────────────────────────────


def scale_reconfig_time(
    configs: list[AcceleratorConfig],
    multiplier: float,
) -> list[AcceleratorConfig]:
    """Create configs with scaled reconfiguration times.

    Args:
        configs: Base accelerator configurations.
        multiplier: Scale factor for reconfig_time_ms (e.g., 0.5 = halved).

    Returns:
        New list of AcceleratorConfig with scaled reconfig times.
    """
    return [
        AcceleratorConfig(
            config_id=c.config_id,
            name=c.name,
            compute_throughput=c.compute_throughput,
            memory_bandwidth=c.memory_bandwidth,
            energy_per_op=c.energy_per_op,
            reconfig_time_ms=c.reconfig_time_ms * multiplier,
            optimal_for=c.optimal_for,
        )
        for c in configs
    ]


# ─── Agent runner ───────────────────────────────────────────────────────


@dataclass
class AgentResult:
    """Result from running a single agent on a workload."""

    agent: str
    avg_reward: float
    total_time_ms: float
    total_energy_mj: float
    cache_hit_rate: float
    n_traces: int
    config_usage: dict[str, int]


def _build_middleware(
    agent: str,
    configs: list[AcceleratorConfig],
    energy_weight: float = 0.15,
    seed: int = 42,
    cache_capacity: int = 2,
) -> AdaptiveMiddleware:
    """Build middleware for a given agent type.

    Supports the same agents as BenchmarkRunner.
    """
    if agent.startswith("static_"):
        config_id = int(agent.split("_")[1])
        mw = AdaptiveMiddleware(
            configs=configs,
            cache_capacity=cache_capacity,
            energy_weight=energy_weight,
        )
        return _wrap_static(mw, config_id)

    if agent == "tabular":
        return AdaptiveMiddleware(
            configs=configs,
            cache_capacity=cache_capacity,
            learning_rate=0.25,
            epsilon_start=0.3,
            energy_weight=energy_weight,
        )

    if agent == "neural":
        mw = AdaptiveMiddleware(
            configs=configs,
            cache_capacity=cache_capacity,
            energy_weight=energy_weight,
        )
        mw.agent = NeuralReconfigAgent(
            configs=configs,
            learning_rate=0.005,
            epsilon_start=0.3,
            energy_weight=energy_weight,
        )
        return mw

    if agent == "random":
        return _wrap_random(configs, seed, energy_weight)

    if agent == "ucb":
        mw = AdaptiveMiddleware(
            configs=configs,
            cache_capacity=cache_capacity,
            energy_weight=energy_weight,
        )
        mw.agent = UCBAgent(
            configs=configs,
            energy_weight=energy_weight,
        )
        return mw

    if agent == "ucb_cache":
        mw = AdaptiveMiddleware(
            configs=configs,
            cache_capacity=cache_capacity,
            energy_weight=energy_weight,
        )
        mw.agent = UCBAgent(
            configs=configs,
            energy_weight=energy_weight,
            cache_aware=True,
        )
        return mw

    if agent == "profile":
        return _wrap_profile(configs, energy_weight, commit_epsilon=0.02)

    if agent == "smart_static":
        return _wrap_profile(configs, energy_weight, commit_epsilon=0.0)

    if agent == "oracle":
        return _wrap_oracle(configs, energy_weight)

    if agent == "lookahead":
        return _wrap_lookahead(configs, energy_weight)

    raise ValueError(f"Unknown agent: {agent!r}")


def run_agent_on_traces(
    agent: str,
    traces: list[WorkloadTrace],
    energy_weight: float = 0.15,
    reconfig_multiplier: float = 1.0,
    seed: int = 42,
    use_cgra: bool = False,
) -> AgentResult:
    """Run a single agent on traces and return results.

    Args:
        agent: Agent name (tabular, neural, oracle, static_N, etc.).
        traces: Workload traces to process.
        energy_weight: Reward function tradeoff (0=throughput, 1=energy).
        reconfig_multiplier: Scale factor for reconfiguration time.
        seed: Random seed for stochastic agents.
        use_cgra: Use CGRA accelerator configs instead of FPGA.

    Returns:
        AgentResult with key metrics.
    """
    base_configs = CGRA_PRESETS if use_cgra else CONFIG_PRESETS
    cache_capacity = CGRA_CACHE_CAPACITY if use_cgra else 2
    configs = scale_reconfig_time(base_configs, reconfig_multiplier)
    mw = _build_middleware(agent, configs, energy_weight, seed, cache_capacity)
    report = mw.run_episode(traces)

    return AgentResult(
        agent=agent,
        avg_reward=report.avg_reward,
        total_time_ms=report.total_time_ms,
        total_energy_mj=report.total_energy_mj,
        cache_hit_rate=report.cache_hit_rate,
        n_traces=report.total_steps,
        config_usage={str(k): v for k, v in report.config_usage.items()},
    )


def run_all_agents_on_traces(
    traces: list[WorkloadTrace],
    energy_weight: float = 0.15,
    reconfig_multiplier: float = 1.0,
    seed: int = 42,
    agents: list[str] | None = None,
    use_cgra: bool = False,
) -> dict[str, AgentResult]:
    """Run all agents on the same traces for comparison.

    Returns a dict of {agent_name: AgentResult}.
    """
    if agents is None:
        agents = [
            "lookahead", "oracle", "smart_static",
            "static_2", "static_3",
            "tabular", "neural", "profile", "ucb", "ucb_cache",
            "random",
        ]

    results: dict[str, AgentResult] = {}
    for agent in agents:
        results[agent] = run_agent_on_traces(
            agent, traces, energy_weight, reconfig_multiplier, seed, use_cgra=use_cgra,
        )

    return results


# ─── Oracle gap computation ────────────────────────────────────────────


def compute_oracle_gap(
    agent_reward: float,
    best_static_reward: float,
    oracle_reward: float,
) -> float | None:
    """Compute how much of the available gain the agent captures.

    Returns a value in [0, 1] where:
        0 = agent == best static (no improvement)
        1 = agent == oracle (optimal)
        <0 = agent is worse than static
        >1 = agent beats oracle (shouldn't happen, but numerical)

    Returns None if oracle == static (no headroom) or if oracle <= best_static
    (the naive oracle doesn't account for reconfiguration cost, so it's not
    a valid upper bound).
    """
    headroom = oracle_reward - best_static_reward
    if abs(headroom) < 1e-9 or headroom < 0:
        return None  # No headroom or oracle is not a valid upper bound
    return (agent_reward - best_static_reward) / headroom


# ─── Load existing results ─────────────────────────────────────────────


def load_existing_results(
    results_dir: str = "benchmarks/results",
) -> list[dict]:
    """Load all existing benchmark JSON results."""
    results: list[dict] = []
    path = Path(results_dir)
    if not path.exists():
        return results

    for fp in sorted(path.glob("*.json")):
        try:
            with open(fp) as f:
                results.append(json.load(f))
        except Exception:
            pass

    return results


# ─── Middleware wrappers (ported from runner.py) ────────────────────────


class _StaticWrapper(AdaptiveMiddleware):
    """Fixed-config baseline."""

    def __init__(self, base: AdaptiveMiddleware, config_id: int) -> None:
        self.__dict__.update(base.__dict__)
        self._static_config_id = config_id

    def run_episode(self, traces, tenant_id: str = "default"):
        return self.run_static_baseline(traces, self._static_config_id)


def _wrap_static(base: AdaptiveMiddleware, config_id: int) -> AdaptiveMiddleware:
    wrapper = _StaticWrapper.__new__(_StaticWrapper)
    wrapper.__dict__.update(base.__dict__)
    wrapper._static_config_id = config_id
    wrapper.__class__ = _StaticWrapper
    return wrapper


class _RandomWrapper(AdaptiveMiddleware):
    """Random config baseline."""

    def __init__(self, configs: list[AcceleratorConfig], seed: int = 42, energy_weight: float = 0.15) -> None:
        super().__init__(configs=configs, energy_weight=energy_weight)
        self._rng = np.random.RandomState(seed)

    def run_episode(self, traces, tenant_id: str = "default"):
        self.reset()
        from adaptive_firmware.runtime.middleware import StepLog
        logs: list[StepLog] = []
        for step, trace in enumerate(traces):
            action = int(self._rng.randint(len(self.configs)))
            result = self.simulator.execute(
                flops=trace.flops, memory_bytes=trace.memory_bytes, config_id=action,
            )
            config = self.simulator.configs[action]
            logs.append(StepLog(
                step=step, tenant_id=tenant_id,
                op_type=trace.op_type, workload_class=trace.workload_class,
                selected_config=action, config_name=config.name,
                exec_time_ms=result.exec_time_ms,
                reconfig_time_ms=result.reconfig_time_ms,
                total_time_ms=result.total_time_ms,
                energy_mj=result.energy_mj,
                reward=0.0,
                cache_hit=result.cache_hit, drift_detected=False, epsilon=0.0,
            ))
        return self._build_report(logs)


def _wrap_random(
    configs: list[AcceleratorConfig],
    seed: int = 42,
    energy_weight: float = 0.15,
) -> AdaptiveMiddleware:
    mw = _RandomWrapper.__new__(_RandomWrapper)
    AdaptiveMiddleware.__init__(mw, configs=configs, energy_weight=energy_weight)
    mw._rng = np.random.RandomState(seed)
    mw.__class__ = _RandomWrapper
    return mw


class _ProfileWrapper(AdaptiveMiddleware):
    """Profile-then-commit agent wrapper."""

    def __init__(self, configs: list[AcceleratorConfig], energy_weight: float = 0.15, commit_epsilon: float = 0.02) -> None:
        super().__init__(configs=configs, energy_weight=energy_weight)
        self.profile_agent = ProfileThenCommitAgent(
            configs=configs,
            profile_steps=10,
            commit_epsilon=commit_epsilon,
        )

    def run_episode(self, traces, tenant_id: str = "default"):
        from adaptive_firmware.observation.telemetry import TelemetryVector
        from adaptive_firmware.runtime.middleware import StepLog
        self.reset()
        self.profile_agent.reset()
        logs: list[StepLog] = []
        for step, trace in enumerate(traces):
            telemetry = self._build_telemetry(trace, tenant_id)
            action = self.profile_agent.select_action(telemetry)
            result = self.simulator.execute(
                flops=trace.flops, memory_bytes=trace.memory_bytes, config_id=action,
            )
            reward = self.profile_agent.compute_reward_for_result(result, self.energy_budget)
            self.profile_agent.update(telemetry, action, reward)
            config = self.simulator.configs[action]
            logs.append(StepLog(
                step=step, tenant_id=tenant_id,
                op_type=trace.op_type, workload_class=trace.workload_class,
                selected_config=action, config_name=config.name,
                exec_time_ms=result.exec_time_ms,
                reconfig_time_ms=result.reconfig_time_ms,
                total_time_ms=result.total_time_ms,
                energy_mj=result.energy_mj,
                reward=reward,
                cache_hit=result.cache_hit, drift_detected=False, epsilon=0.0,
            ))
        return self._build_report(logs)


def _wrap_profile(
    configs: list[AcceleratorConfig],
    energy_weight: float = 0.15,
    commit_epsilon: float = 0.02,
) -> AdaptiveMiddleware:
    mw = _ProfileWrapper.__new__(_ProfileWrapper)
    AdaptiveMiddleware.__init__(mw, configs=configs, energy_weight=energy_weight)
    mw.profile_agent = ProfileThenCommitAgent(
        configs=configs,
        profile_steps=10,
        commit_epsilon=commit_epsilon,
    )
    mw.__class__ = _ProfileWrapper
    return mw


class _OracleWrapper(AdaptiveMiddleware):
    """Perfect-knowledge oracle baseline."""

    _OPTIMAL = {
        "compute_bound": 0,
        "memory_bound": 1,
        "balanced": 2,
    }

    def run_episode(self, traces, tenant_id: str = "default"):
        from adaptive_firmware.observation.telemetry import TelemetryVector
        from adaptive_firmware.runtime.middleware import StepLog
        self.reset()
        # Pre-fill Q-table
        import numpy as np
        for state, best_action in self._OPTIMAL.items():
            for a in range(len(self.configs)):
                self.agent.q_table[state][a] = 1.0 if a == best_action else 0.0

        logs: list[StepLog] = []
        for step, trace in enumerate(traces):
            telemetry = self._build_telemetry(trace, tenant_id)
            action = self._OPTIMAL.get(trace.workload_class, 2)
            result = self.simulator.execute(
                flops=trace.flops, memory_bytes=trace.memory_bytes, config_id=action,
            )
            reward = self.agent.compute_reward(result, self.energy_budget)
            self.agent.update(telemetry, action, reward)
            config = self.simulator.configs[action]
            logs.append(StepLog(
                step=step, tenant_id=tenant_id,
                op_type=trace.op_type, workload_class=trace.workload_class,
                selected_config=action, config_name=config.name,
                exec_time_ms=result.exec_time_ms,
                reconfig_time_ms=result.reconfig_time_ms,
                total_time_ms=result.total_time_ms,
                energy_mj=result.energy_mj,
                reward=reward,
                cache_hit=result.cache_hit, drift_detected=False, epsilon=0.0,
            ))
        return self._build_report(logs)


def _wrap_oracle(
    configs: list[AcceleratorConfig],
    energy_weight: float = 0.15,
) -> AdaptiveMiddleware:
    mw = _OracleWrapper.__new__(_OracleWrapper)
    AdaptiveMiddleware.__init__(mw, configs=configs, energy_weight=energy_weight)
    # Pre-fill Q-table
    for state, best_action in _OracleWrapper._OPTIMAL.items():
        for a in range(len(configs)):
            mw.agent.q_table[state][a] = 1.0 if a == best_action else 0.0
    mw.__class__ = _OracleWrapper
    return mw


def _wrap_lookahead(
    configs: list[AcceleratorConfig],
    energy_weight: float = 0.15,
) -> AdaptiveMiddleware:
    """Create a look-ahead oracle middleware.

    The look-ahead oracle uses dynamic programming over the full trace
    sequence with a cache-aware state model (2-slot LRU). This is the
    true upper bound — it accounts for reconfiguration cost while
    exploiting the cache to eliminate switch overhead between cached configs.
    """
    from benchmarks.analysis.lookahead_oracle import LookAheadOracleMiddleware as _LAM
    mw = _LAM.__new__(_LAM)
    AdaptiveMiddleware.__init__(mw, configs=configs, energy_weight=energy_weight)
    mw.energy_weight = energy_weight
    mw.__class__ = _LAM
    return mw
