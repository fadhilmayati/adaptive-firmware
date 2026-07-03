# Hardware that learns

**The story:** [Read the post →](STATUS.md) — what I built, what I learned, and why it matters.

**The original idea:** [The essay that started it all →](SOURCE_CLAIM.md)

**For the technical crowd:** [The verification trace →](concepts/workload-adaptive-silicon-verification.md) and the code itself below.

---

## What is this?

A small system that watches what your AI code is doing and reconfigures the chip underneath to match. It learns as it goes, so it gets better the more you use it.

It runs on my laptop. It beats the alternative by 18.8% on realistic workloads. And the code is all open source.

## Why should you care?

If you've ever wondered why your AI app is slow, or burns through power, or costs too much to run — some of that is the hardware being stuck with a design that was locked in years before your workload existed.

This is a way to fix that. Not in five years. Now, in simulation, and soon on a $249 dev board.

## What's in the box

If you want to dig into the code:

```
src/
├── hardware/        The simulated chip
├── observation/     How it watches your AI code
├── agent/           The brain that makes decisions
├── runtime/         The glue that holds it together
└── workloads/       Test scenarios
```

Four demo scripts, 36 tests, all passing. You can run the whole thing on a MacBook in seconds.

```bash
pip install -e .

# See it learn in real time
python scripts/run_demo.py

# Test it on realistic AI workloads
python scripts/run_real_benchmarks.py

# Compare different learning approaches
python scripts/compare_agents.py

# See how look-ahead scheduling helps
python scripts/run_lookahead_demo.py
```

## What's the catch?

This is a proof of concept, not a product. It works in simulation. It hasn't been on real hardware yet. And it's best for mixed, unpredictable workloads — if your AI is doing one thing over and over, the simple approach is probably fine.

But the loop is real, the learning is real, and the gap in the market is real.

## Let's talk

I'm one person with a laptop. The space is huge. If any of this is interesting to you, **[let's catch up →](mailto:muhammadfadhilmayati@gmail.com)**.

I'd especially love to hear from:
- People running AI at scale who feel the hardware bottleneck
- Hardware folks with war stories from the reconfigurable chip world
- Investors tracking AI infrastructure
- Researchers who want to collaborate

No pitch. No deck. Just a conversation.

---

*The full technical verification — claim by claim, source by source — is at [concepts/workload-adaptive-silicon-verification.md](concepts/workload-adaptive-silicon-verification.md) for anyone who wants to go deep.*