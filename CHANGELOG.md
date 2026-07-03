# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Marketing-friendly STATUS.md with "let's catch up" CTA
- SOURCE_CLAIM.md preserving the original essay
- GitHub-ready repository setup (LICENSE, CI, issue templates, CONTRIBUTING)
- concepts/workload-adaptive-silicon-verification.md with full claim-by-claim trace
- PyTorch integration via forward hooks
- Real workload benchmarks (LLM token streaming, YOLO, Whisper, mixed production)
- Neural network agent (small MLP with experience replay and target network)
- Look-ahead scheduling with prefetching (reduces reconfig overhead by 21%)
- Multi-tenant workload arbitration
- ADWIN-inspired concept drift detection
- Informed initialization (workload-class priors)

### Changed
- Reward function: increased cache-hit bonus and reconfig penalty based on real benchmark results

### Results
- 36/36 tests passing
- Adaptive agent beats best static config by 18.8% on mixed production workload (3268 ops)
- Adaptive agent beats best static by 2.2% on LLM token streaming (328 ops)
- Look-ahead scheduling reduces reconfig overhead by 21%

## [0.1.0] - 2026-07-02

### Added
- Initial proof-of-concept release
- Hardware simulator with 4 reconfigurable accelerator configs (roofline model)
- LRU bitstream cache
- Tabular Q-learning RL agent with epsilon-greedy exploration
- ADWIN-inspired concept drift detector
- PyTorch observer with forward hooks (Conv2d, Linear, MultiheadAttention)
- Multi-tenant middleware
- 24 tests (hardware simulator, agent, middleware, e2e)
- Four demo scripts (basic, multi-tenant, static baseline, comparison)

### Results
- Adaptive agent outperforms best static config on mixed workloads
- 36% learning improvement from early exploration to late exploitation
- Multi-tenant demo shows different configs being used for different tenants
