"""Benchmark runner.

The runner takes a workload and an agent configuration, runs the
adaptive middleware on the workload, and produces a structured
result that can be compared across agents and workloads.

Results are written as JSON to the results/ directory. The format
is standardized so anyone can run the same benchmarks and submit
comparable results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .workloads.base import WorkloadSpec
from .workloads.registry import get_workload
from adaptive_firmware.hardware.configs import CONFIG_PRESETS, AcceleratorConfig
from adaptive_firmware.runtime.middleware import AdaptiveMiddleware
from adaptive_firmware.agent.rl_agent import ReconfigAgent
from adaptive_firmware.agent.neural_agent import NeuralReconfigAgent
from adaptive_firmware.agent.profile_agent import ProfileThenCommitAgent


@dataclass
class BenchmarkResult:
    """Standardized result from running a (workload, agent) combination.

    This is the format everyone submits. Adding fields is fine (additive),
    but renaming or removing fields requires a suite version bump.
    """

    suite_version: str
    timestamp: str
    workload_name: str
    workload_version: str
    agent_name: str
    agent_config: dict
    n_traces: int
    avg_reward: float
    total_time_ms: float
    total_energy_mj: float
    total_reconfig_time_ms: float
    cache_hit_rate: float
    config_usage: dict[str, int]
    final_policy: dict[str, int]
    run_time_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class BenchmarkRunner:
    """Runs benchmarks and writes structured results.

    Usage:
        runner = BenchmarkRunner(output_dir="benchmarks/results")
        result = runner.run("llm_decode", agent="tabular")
    """

    def __init__(self, output_dir: str = "benchmarks/results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.suite_version = "0.1.0"

    def run(
        self,
        workload_name: str,
        agent: str = "tabular",
        agent_config: dict | None = None,
        workload_version: str | None = None,
    ) -> BenchmarkResult:
        """Run a single benchmark and return the result.

        Args:
            workload_name: Name of the workload (e.g., "llm_decode").
            agent: Agent type — "tabular", "neural", "static_0", etc.
            agent_config: Optional agent-specific config overrides.
            workload_version: Optional specific workload version.

        Returns:
            BenchmarkResult with all metrics.
        """
        spec = get_workload(workload_name, workload_version)
        traces = spec.generate()

        if not spec.validate(traces, tolerance=0.3):
            raise ValueError(
                f"Workload {spec.name}@{spec.version} produced invalid traces "
                f"(got {len(traces)}, expected ~{spec.expected_n_traces})"
            )

        start = time.time()
        mw = self._build_middleware(agent, agent_config or {})
        report = mw.run_episode(traces)
        run_time = time.time() - start

        result = BenchmarkResult(
            suite_version=self.suite_version,
            timestamp=datetime.now().isoformat() + "Z",
            workload_name=spec.name,
            workload_version=spec.version,
            agent_name=agent,
            agent_config=agent_config or {},
            n_traces=report.total_steps,
            avg_reward=report.avg_reward,
            total_time_ms=report.total_time_ms,
            total_energy_mj=report.total_energy_mj,
            total_reconfig_time_ms=report.total_reconfig_time_ms,
            cache_hit_rate=report.cache_hit_rate,
            config_usage={str(k): v for k, v in report.config_usage.items()},
            final_policy={k: v for k, v in report.final_policy.items()},
            run_time_seconds=run_time,
        )

        self._save(result)
        return result

    def _build_middleware(self, agent: str, config: dict) -> AdaptiveMiddleware:
        """Build a middleware instance configured for the given agent."""
        if agent.startswith("static_"):
            config_id = int(agent.split("_")[1])
            mw = AdaptiveMiddleware(
                configs=CONFIG_PRESETS,
                cache_capacity=config.get("cache_capacity", 2),
            )
            return _StaticConfigMiddleware(mw, config_id)

        if agent == "tabular":
            return AdaptiveMiddleware(
                configs=CONFIG_PRESETS,
                cache_capacity=config.get("cache_capacity", 2),
                learning_rate=config.get("learning_rate", 0.25),
                epsilon_start=config.get("epsilon_start", 0.3),
            )

        if agent == "neural":
            return AdaptiveMiddleware(
                configs=CONFIG_PRESETS,
                cache_capacity=config.get("cache_capacity", 2),
            )
            # Replace the agent with a neural one
            mw.agent = NeuralReconfigAgent(
                configs=CONFIG_PRESETS,
                learning_rate=config.get("learning_rate", 0.005),
                epsilon_start=config.get("epsilon_start", 0.3),
            )

        if agent == "random":
            return _RandomConfigMiddleware(
                configs=CONFIG_PRESETS,
                seed=config.get("seed", 42),
            )

        if agent == "profile":
            return _ProfileMiddleware(
                configs=CONFIG_PRESETS,
                profile_steps=config.get("profile_steps", 10),
            )

        if agent == "oracle":
            return _OracleMiddleware(
                configs=CONFIG_PRESETS,
            )

        raise ValueError(f"Unknown agent: {agent!r}. Use tabular, neural, static_N, random, or oracle.")

    def _save(self, result: BenchmarkResult) -> None:
        """Save result to JSON file."""
        filename = f"{result.workload_name}_{result.workload_version}_{result.agent_name}_{result.timestamp.replace(':', '-')}.json"
        path = self.output_dir / filename
        path.write_text(result.to_json())


class _StaticConfigMiddleware(AdaptiveMiddleware):
    """Middleware that uses a fixed config (no adaptation)."""

    def __init__(self, base: AdaptiveMiddleware, config_id: int) -> None:
        # Copy state but keep the same simulator
        self.__dict__.update(base.__dict__)
        self._static_config_id = config_id

    def run_episode(self, traces, tenant_id: str = "default"):
        return self.run_static_baseline(traces, self._static_config_id)


class _RandomConfigMiddleware(AdaptiveMiddleware):
    """Middleware that picks a random config each step."""

    def __init__(self, configs: list[AcceleratorConfig], seed: int = 42) -> None:
        super().__init__(configs=configs)
        import numpy as np
        self._rng = np.random.RandomState(seed)

    def run_episode(self, traces, tenant_id: str = "default"):
        from adaptive_firmware.runtime.middleware import StepLog, EpisodeReport
        from adaptive_firmware.hardware.configs import CONFIG_PRESETS
        self.reset()
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
                reward=0.0,  # No learning
                cache_hit=result.cache_hit, drift_detected=False, epsilon=0.0,
            ))
        return self._build_report(logs)


class _ProfileMiddleware(AdaptiveMiddleware):
    """Middleware that uses the profile-then-commit agent.

    Profiles each config for `profile_steps` traces, then commits to
    the best-performing config for the rest of the workload. This is
    a production-realistic policy: bounded profiling cost, minimal
    reconfiguration during commit, drift-triggered re-profiling.
    """

    def __init__(self, configs: list[AcceleratorConfig], profile_steps: int = 10) -> None:
        super().__init__(configs=configs)
        self.profile_agent = ProfileThenCommitAgent(
            configs=configs,
            profile_steps=profile_steps,
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
                cache_hit=result.cache_hit,
                drift_detected=False,
                epsilon=0.0,
            ))
        return self._build_report(logs)


class _OracleMiddleware(AdaptiveMiddleware):
    """Middleware that always picks the best config for the workload class.

    The oracle knows the optimal mapping (workload_class → best config)
    and uses it. This is the upper bound on what any adaptive agent
    can achieve.
    """

    _OPTIMAL = {
        "compute_bound": 0,  # HIGH_COMPUTE
        "memory_bound": 1,   # HIGH_BANDWIDTH
        "balanced": 2,       # BALANCED
    }

    def __init__(self, configs: list[AcceleratorConfig]) -> None:
        super().__init__(configs=configs)
        # Override the agent with a "perfect" policy
        self.agent._last_state = "unknown"
        # Pre-fill Q-table with optimal values
        import numpy as np
        for state, best_action in self._OPTIMAL.items():
            for a in range(len(configs)):
                self.agent.q_table[state][a] = 1.0 if a == best_action else 0.0

    def run_episode(self, traces, tenant_id: str = "default"):
        from adaptive_firmware.observation.telemetry import TelemetryVector
        from adaptive_firmware.runtime.middleware import StepLog
        self.reset()
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


def run_workload(workload_name: str, **kwargs) -> BenchmarkResult:
    """Convenience function: run a single benchmark."""
    return BenchmarkRunner().run(workload_name, **kwargs)


def run_all(output_dir: str = "benchmarks/results") -> list[BenchmarkResult]:
    """Run all workloads with all agents.

    This is the full benchmark suite — the canonical results everyone
    can compare against.
    """
    from .workloads.registry import list_workloads

    runner = BenchmarkRunner(output_dir=output_dir)
    results: list[BenchmarkResult] = []

    agents = ["oracle", "static_2", "static_3", "tabular", "neural", "profile", "random"]
    for spec in list_workloads():
        for agent in agents:
            print(f"  Running {spec.name}@{spec.version} with {agent}...")
            result = runner.run(spec.name, agent=agent)
            results.append(result)

    return results
