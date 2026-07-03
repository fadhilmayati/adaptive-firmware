# Contributing

Thanks for being here. This project is at the proof-of-concept stage, and contributions that make it more useful, more correct, or more accessible are very welcome.

## Where to start

The easiest way to get oriented is to run the demos:

```bash
git clone <repo-url>
cd adaptive-firmware
pip install -e ".[dev]"
python scripts/run_demo.py
python scripts/run_real_benchmarks.py
```

Then read [`STATUS.md`](STATUS.md) for the story and [`concepts/workload-adaptive-silicon-verification.md`](concepts/workload-adaptive-silicon-verification.md) for the source-by-source verification of the underlying claims.

## Good first contributions

Here are some things that would be genuinely useful, in rough order of effort:

**Quick (< 1 hour)**
- Add a new accelerator config to `src/adaptive_firmware/hardware/configs.py` and verify the benchmarks still pass
- Add a new workload pattern to `src/adaptive_firmware/workloads/real_models.py`
- Improve docstrings on any public function

**Medium (1-4 hours)**
- Add a new RL agent (e.g., DQN, PPO, or a Bayesian bandit) to `src/adaptive_firmware/agent/` and benchmark it against the existing ones
- Add a richer workload classifier (currently uses arithmetic intensity; could use memory access patterns, tensor shapes, etc.)
- Add a real PyTorch op that the observer can profile (currently supports Conv2d, Linear, MultiheadAttention)

**Larger (4+ hours)**
- Port the simulator to a proper cycle-accurate CGRA model (e.g., sarchlab/zeonica)
- Add a Kria KV260 hardware backend (real PR via Linux FPGA Manager)
- Add a standardized benchmark suite that other researchers can reproduce

## Style guide

A few things to keep in mind:

- **Type hints** on all public functions
- **Docstrings** on all public classes and functions (Google style)
- **Tests** for any new code — we have 36 tests and we want to keep that number growing
- **No new top-level dependencies** without discussion (we want this to run on a MacBook CPU)
- **No `as any` or type suppression** — keep the type system honest

## Running the tests

```bash
python -m pytest tests/ -v          # full test suite
python -m pytest tests/ -v -k "agent"  # just agent tests
python -m pytest tests/ --tb=short   # shorter tracebacks
```

## Before opening a PR

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Docstrings updated
- [ ] The relevant demo script still runs without errors

## Code of conduct

Be kind. Assume good faith. Disagree on ideas, not people. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Questions?

Open an issue, or — if you'd rather talk before writing code — reach out directly. The contact info is at the bottom of [`STATUS.md`](STATUS.md).
