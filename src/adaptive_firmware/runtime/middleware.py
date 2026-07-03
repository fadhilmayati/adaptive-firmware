"""Adaptive middleware: the core loop that connects AI runtime to silicon.

This is the product — the layer that sits between PyTorch and the
reconfigurable hardware, closing the loop:

    observe workload → decide config → execute on hardware → measure → learn

The middleware processes workload traces from the PyTorchObserver,
feeds them to the RL agent for config decisions, executes on the
HardwareSimulator, computes rewards, and feeds them back to the agent.

For multi-tenant support, the middleware can handle multiple workload
streams, each with its own model and its own set of traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..hardware.simulator import HardwareSimulator, ExecutionResult
from ..hardware.configs import AcceleratorConfig, CONFIG_PRESETS
from ..observation.telemetry import TelemetryVector, WorkloadTrace
from ..observation.pytorch_hooks import PyTorchObserver
from ..agent.rl_agent import ReconfigAgent


@dataclass
class StepLog:
    """Log entry for a single middleware step."""

    step: int
    tenant_id: str
    op_type: str
    workload_class: str
    selected_config: int
    config_name: str
    exec_time_ms: float
    reconfig_time_ms: float
    total_time_ms: float
    energy_mj: float
    reward: float
    cache_hit: bool
    drift_detected: bool
    epsilon: float


@dataclass
class EpisodeReport:
    """Summary of a complete middleware episode."""

    total_steps: int
    total_time_ms: float
    total_energy_mj: float
    total_reconfig_time_ms: float
    avg_reward: float
    cache_hit_rate: float
    drift_count: int
    final_policy: dict[str, int]
    logs: list[StepLog] = field(default_factory=list)
    config_usage: dict[int, int] = field(default_factory=dict)

    def summary_str(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Steps: {self.total_steps}",
            f"Total time: {self.total_time_ms:.2f} ms",
            f"Total energy: {self.total_energy_mj:.2f} mJ",
            f"Reconfig overhead: {self.total_reconfig_time_ms:.2f} ms "
            f"({self.total_reconfig_time_ms/max(self.total_time_ms,1e-9)*100:.1f}%)",
            f"Avg reward: {self.avg_reward:.4f}",
            f"Cache hit rate: {self.cache_hit_rate*100:.1f}%",
            f"Drift events: {self.drift_count}",
            f"Final policy: {self.final_policy}",
        ]
        return "\n  ".join(lines)


class AdaptiveMiddleware:
    """The adaptive firmware layer.

    Orchestrates the observe-decide-execute-learn loop. Can run in
    single-tenant or multi-tenant mode.

    Usage (single-tenant):
        mw = AdaptiveMiddleware(configs=CONFIG_PRESETS)
        report = mw.run_episode(traces)

    Usage (multi-tenant):
        mw = AdaptiveMiddleware(configs=CONFIG_PRESETS)
        report = mw.run_multi_tenant({
            "cnn_model": cnn_traces,
            "transformer": transformer_traces,
        })
    """

    def __init__(
        self,
        configs: list[AcceleratorConfig] | None = None,
        cache_capacity: int = 2,
        energy_weight: float | None = None,
        learning_rate: float = 0.1,
        epsilon_start: float = 0.3,
        verbose: bool = False,
    ) -> None:
        """Initialize the middleware.

        Args:
            configs: Accelerator configs. Defaults to CONFIG_PRESETS.
            cache_capacity: Number of PRRs in the bitstream cache.
            energy_weight: Weight for energy in reward computation.
            learning_rate: RL agent learning rate.
            epsilon_start: Initial exploration rate.
            verbose: Print step-by-step logs.
        """
        self.configs = configs or CONFIG_PRESETS
        self.simulator = HardwareSimulator(self.configs, cache_capacity=cache_capacity)
        self.agent = ReconfigAgent(
            configs=self.configs,
            learning_rate=learning_rate,
            epsilon_start=epsilon_start,
            energy_weight=energy_weight,
        )
        self.verbose = verbose
        self.energy_budget = 1.0  # Start with full budget
        self.total_energy_consumed_mj = 0.0

    def _build_telemetry(
        self,
        trace: WorkloadTrace,
        tenant_id: str = "default",
    ) -> TelemetryVector:
        """Build a TelemetryVector from a workload trace."""
        return TelemetryVector(
            op_type=trace.op_type,
            flops=trace.flops,
            memory_bytes=trace.memory_bytes,
            arithmetic_intensity=trace.arithmetic_intensity,
            workload_class=trace.workload_class,
            current_config_id=self.simulator.current_config_id,
            cache_loaded_configs=self.simulator.cache.loaded_config_ids,
            energy_budget_remaining=self.energy_budget,
            latency_target_ms=50.0,  # 50ms SLO target for PoC
        )

    def _process_trace(
        self,
        trace: WorkloadTrace,
        step: int,
        tenant_id: str = "default",
    ) -> StepLog:
        """Process a single workload trace through the full loop."""
        # 1. Observe: build telemetry from workload + hardware state
        telemetry = self._build_telemetry(trace, tenant_id)

        # 2. Decide: agent selects a config
        action = self.agent.select_action(telemetry)
        config = self.agent.config_by_id[action]

        # 3. Execute: run on simulated hardware
        result = self.simulator.execute(
            flops=trace.flops,
            memory_bytes=trace.memory_bytes,
            config_id=action,
        )

        # 4. Measure: compute reward
        reward = self.agent.compute_reward(result, self.energy_budget)

        # 5. Learn: update agent
        self.agent.update(telemetry, action, reward)

        # Track energy budget (deplete as we consume)
        self.total_energy_consumed_mj += result.energy_mj
        # Budget depletes slowly (normalized to episode length)
        # Reset handled by reset()

        log = StepLog(
            step=step,
            tenant_id=tenant_id,
            op_type=trace.op_type,
            workload_class=trace.workload_class,
            selected_config=action,
            config_name=config.name,
            exec_time_ms=result.exec_time_ms,
            reconfig_time_ms=result.reconfig_time_ms,
            total_time_ms=result.total_time_ms,
            energy_mj=result.energy_mj,
            reward=reward,
            cache_hit=result.cache_hit,
            drift_detected=self.agent.drift_detector.drift_detected,
            epsilon=self.agent.epsilon,
        )

        if self.verbose:
            print(
                f"  [{tenant_id}] step={step:4d} {trace.op_type:10s} "
                f"({trace.workload_class:14s}) -> {config.name:14s} "
                f"exec={result.exec_time_ms:8.3f}ms reconfig={result.reconfig_time_ms:5.1f}ms "
                f"reward={reward:.3f} eps={self.agent.epsilon:.3f}"
                + (" DRIFT!" if self.agent.drift_detector.drift_detected else "")
            )

        return log

    def run_episode(
        self,
        traces: list[WorkloadTrace],
        tenant_id: str = "default",
    ) -> EpisodeReport:
        """Run a single-tenant episode over a list of workload traces.

        Args:
            traces: Workload traces to process (in order).
            tenant_id: Identifier for this tenant.

        Returns:
            EpisodeReport with full results.
        """
        self.reset()
        logs: list[StepLog] = []

        for step, trace in enumerate(traces):
            log = self._process_trace(trace, step, tenant_id)
            logs.append(log)

        return self._build_report(logs)

    def run_multi_tenant(
        self,
        tenant_traces: dict[str, list[WorkloadTrace]],
        interleave: bool = True,
    ) -> EpisodeReport:
        """Run a multi-tenant episode.

        Workload traces from multiple tenants are interleaved (round-robin
        by default) to simulate concurrent execution. The agent must learn
        to handle the mixed workload — this is the key differentiation from
        prior academic work which is all single-tenant.

        Args:
            tenant_traces: Dict of tenant_id -> list of traces.
            interleave: If True, round-robin between tenants. If False,
                       run all of each tenant's traces sequentially.

        Returns:
            EpisodeReport with full results.
        """
        self.reset()
        logs: list[StepLog] = []
        step = 0

        if interleave:
            # Round-robin interleave
            max_len = max(len(t) for t in tenant_traces.values())
            for i in range(max_len):
                for tid, traces in tenant_traces.items():
                    if i < len(traces):
                        log = self._process_trace(traces[i], step, tid)
                        logs.append(log)
                        step += 1
        else:
            for tid, traces in tenant_traces.items():
                for trace in traces:
                    log = self._process_trace(trace, step, tid)
                    logs.append(log)
                    step += 1

        return self._build_report(logs)

    def run_static_baseline(
        self,
        traces: list[WorkloadTrace],
        config_id: int,
        tenant_id: str = "static",
    ) -> EpisodeReport:
        """Run a static configuration baseline (no agent, fixed config).

        Used for comparison: the agent should outperform any single
        static configuration when workloads are mixed.
        """
        self.simulator.reset()
        logs: list[StepLog] = []
        config = self.simulator.configs[config_id]

        for step, trace in enumerate(traces):
            result = self.simulator.execute(
                flops=trace.flops,
                memory_bytes=trace.memory_bytes,
                config_id=config_id,
            )

            # Compute what the agent's reward would have been (for comparison)
            reward = self.agent.compute_reward(result, 1.0)

            logs.append(StepLog(
                step=step,
                tenant_id=tenant_id,
                op_type=trace.op_type,
                workload_class=trace.workload_class,
                selected_config=config_id,
                config_name=config.name,
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

    def _build_report(self, logs: list[StepLog]) -> EpisodeReport:
        """Build an EpisodeReport from step logs."""
        total_steps = len(logs)
        total_time = sum(l.total_time_ms for l in logs)
        total_energy = sum(l.energy_mj for l in logs)
        total_reconfig = sum(l.reconfig_time_ms for l in logs)
        avg_reward = sum(l.reward for l in logs) / max(total_steps, 1)
        cache_hits = sum(1 for l in logs if l.cache_hit)
        cache_hit_rate = cache_hits / max(total_steps, 1)

        config_usage: dict[int, int] = {}
        for l in logs:
            config_usage[l.selected_config] = config_usage.get(l.selected_config, 0) + 1

        return EpisodeReport(
            total_steps=total_steps,
            total_time_ms=total_time,
            total_energy_mj=total_energy,
            total_reconfig_time_ms=total_reconfig,
            avg_reward=avg_reward,
            cache_hit_rate=cache_hit_rate,
            drift_count=self.agent.drift_detector.drift_count,
            final_policy=self._get_agent_policy(),
            logs=logs,
            config_usage=config_usage,
        )

    def reset(self) -> None:
        """Reset middleware state for a new episode."""
        self.simulator.reset()
        self.agent.reset()
        self.energy_budget = 1.0
        self.total_energy_consumed_mj = 0.0

    def _get_agent_policy(self) -> dict[str, int]:
        """Get the agent's current policy, handling both tabular and neural agents.

        The tabular agent has get_policy() with no args.
        The neural agent has get_policy_dict() with no args.
        """
        if hasattr(self.agent, "get_policy_dict"):
            return self.agent.get_policy_dict()
        elif hasattr(self.agent, "get_policy"):
            try:
                return self.agent.get_policy()
            except TypeError:
                return {"unknown": 0}
        return {"unknown": 0}

    def collect_traces_from_model(
        self,
        model,
        input_generator: Callable,
        n_runs: int = 10,
    ) -> list[WorkloadTrace]:
        """Run a PyTorch model and collect workload traces.

        Args:
            model: A PyTorch nn.Module.
            input_generator: Callable that returns model input tensors.
            n_runs: Number of forward passes to collect.

        Returns:
            List of WorkloadTrace objects.
        """
        observer = PyTorchObserver(model)
        model.eval()
        with __import__("torch").no_grad():
            for _ in range(n_runs):
                inp = input_generator()
                model(inp)
        traces = observer.pop_traces()
        observer.remove_hooks()
        return traces
