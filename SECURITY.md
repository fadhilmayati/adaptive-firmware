# Security Policy

## Supported Versions

This project is at the proof-of-concept stage. The current development branch
is the only supported version.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < main  | :x:                |

## Reporting a Vulnerability

This is a research project that runs in simulation. There is no production
deployment, no network exposure, and no real hardware. The attack surface is
limited to the Python code itself.

If you discover a security issue in the code (e.g., a dependency with a known
vulnerability, an unsafe `eval`, a path traversal in workload generation),
please report it by:

1. Opening a private issue (if GitHub private issues are enabled)
2. Emailing the maintainer directly (contact info in [STATUS.md](STATUS.md))

Please do **not** open a public issue for security vulnerabilities.

## What to expect

- Acknowledgment within 3 days
- Assessment within 1 week
- Fix timeline communicated after assessment

## Scope

The following are in scope:

- Vulnerabilities in the Python code (`src/adaptive_firmware/`)
- Vulnerabilities in the demo scripts (`scripts/`)
- Vulnerabilities in dependencies listed in `pyproject.toml`
- Unsafe handling of workload data, model inputs, or telemetry

The following are out of scope:

- The simulation results themselves (they're research output, not security boundaries)
- The academic claims (those are scientific, not security)
- The marketing post (it's prose, not code)

## Notes

This project is a proof of concept, not a production system. The security
posture is "appropriate for a research codebase." If you're using it in a
context where security matters, please contact the maintainer first.
