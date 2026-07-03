# Documentation

This directory contains technical documentation for the adaptive firmware
layer. For the high-level story, see [STATUS.md](../STATUS.md). For the
underlying research, see
[concepts/workload-adaptive-silicon-verification.md](../concepts/workload-adaptive-silicon-verification.md).

## Documents

- **[architecture.md](architecture.md)** — How the system is put together:
  module breakdown, data flow, key design decisions, future direction.

## Other documentation in the repo

- **[README.md](../README.md)** — Landing page
- **[STATUS.md](../STATUS.md)** — The story (from claim to working PoC)
- **[SOURCE_CLAIM.md](../SOURCE_CLAIM.md)** — The original essay
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — How to contribute
- **[CHANGELOG.md](../CHANGELOG.md)** — What changed when
- **[CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)** — Community standards
- **[SECURITY.md](../SECURITY.md)** — Security policy

## For the deep technical reader

The full source code is documented with docstrings. The key entry points:

- `src/adaptive_firmware/runtime/middleware.py` — `AdaptiveMiddleware` is the main API
- `src/adaptive_firmware/agent/rl_agent.py` — `ReconfigAgent` is the default learning agent
- `src/adaptive_firmware/hardware/simulator.py` — `HardwareSimulator` is the hardware model

The [architecture.md](architecture.md) document gives a tour of all the
moving parts.
