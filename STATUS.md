# What if your hardware could learn?

*A few months ago I had an idea. Today it's running on my laptop. Here's the story.*

---

## It started with a feeling

I build tools that test chips before they ship. It's the unglamorous part of hardware — the part that happens long before anyone holds a phone or starts a car.

Lately I've been using AI to help me write code. And something kept bugging me.

The AI gets smarter every week. It picks up on how I work. It starts finishing my sentences. After a few months together, it feels less like a tool and more like a really sharp colleague.

But the chip it's running on? That's frozen. Exactly the way it was designed, years ago, before anyone knew what workloads would look like.

We taught software to adapt. We never asked hardware to.

That felt like a missed opportunity.

## So I started asking: what if it could?

What if the chip could watch what's actually happening, and reshape itself to fit? Not a person making that call. The system.

Turns out, this isn't science fiction. There's a whole family of chips that can already do this — they just don't get much attention outside the hardware world:

- **Chips that rewire themselves.** You buy them with one design, then change their mind after they're already in your device.
- **Chips built for AI data flows.** Optimized for the kind of math that powers today's models.
- **Chips that compute inside memory.** No more waiting for data to travel back and forth — the math happens where the data lives.
- **Brain-inspired chips.** Where the hardware itself learns, the way synapses in your brain strengthen or weaken.

The hardware can change. We just haven't built the software that lets it happen automatically.

So I started building it.

## What's working so far

I built a small system that sits between your AI code and the chip underneath. It watches what your model is doing, picks the best chip configuration for the moment, and switches when the workload changes. Then it learns from the result, so it gets better over time.

I tested it against the alternative — just picking one configuration and sticking with it. The honest result: **the adaptive system wins on heterogeneous workloads, but only in a narrow window of the energy-vs-throughput tradeoff.**

The benchmark suite sweeps the energy weight from 0.0 (throughput-only) to 1.0 (energy-only) and reports who wins on each workload:

- **On `mixed_production`** (the real production scenario), the adaptive agent wins at `energy_weight=0.5` — a balanced tradeoff — beating the best static by **+1.9%**. It loses at every other weight.
- **On `llm_decode`** (long-running token streaming), static LOW_POWER wins across all energy weights. The workload is too homogeneous for adaptation to pay off.
- **On short workloads** (vision, audio), static configs win because the exploration cost of adaptation eats the gain.

The gain is real but narrow. It's biggest when the workload keeps changing AND the reward function balances throughput and energy, not at either extreme. For a single, predictable task, the simple approach is fine. For a production AI serving system with mixed traffic and a realistic SLO, the adaptive approach gives you a small but real edge.

I also added a **compile-time baseline** (`smart_static`) that represents what a static compiler like TVM or TensorRT would produce — profile the workload once at startup, commit to the best config, and never explore again. The result: smart_static matches or beats the profile agent on short workloads (where drift is unlikely), but on `mixed_production` the tabular adaptive agent still wins. The gap is narrow (+1.9%) because this workload has long enough phases for profile-then-commit to work well — but the adaptive agent doesn't need to know phase boundaries in advance.

## What I learned along the way

Three things became clear:

1. **The idea is real, and it's not just me.** Three research papers from 2025-2026 already proved the same thing on real hardware. The novelty isn't in the concept — it's in making it open-source, easy to plug into existing AI frameworks, and smart enough to learn while it runs.

2. **The original claim was a bit off.** When I first wrote about this, I said "we never asked hardware to adapt." That's not quite right. Chips that can rewire themselves have been around since 1985. People have been working on this for 40 years. What's new is the *software* to make it automatic and accessible.

3. **There's a real opportunity here.** No open-source project ties AI frameworks to adaptive hardware. The pieces exist separately — the AI runtimes, the reconfiguration tools, the hardware simulators. Nobody has stitched them together with online learning and multi-tenant support. That's the gap.

## Where this goes next

Here's what's on the roadmap:

- [x] Real workload benchmarks (language models, vision, audio) — done
- [x] Neural network policy for the learning brain — done
- [x] Look-ahead scheduling that hides reconfiguration cost — done
- [ ] A more advanced chip type (CGRA) that reconfigures faster
- [ ] A real hardware port — starting with a $249 dev board
- [ ] A standardized benchmark so the field can measure progress

**The full code is open source.** 36 tests passing, runs on a MacBook, and you can see every line.

## Let's catch up

If any of this sounds interesting to you — whether you're deep in hardware, building AI infrastructure, investing in the space, or just curious — I'd genuinely love to chat.

This is the kind of idea that gets better the more people poke at it. I'm one person with a laptop. The space is huge.

If you'd like to know more, **[let's catch up →](mailto:muhammadfadhilmayati@gmail.com)** — coffee, virtual or otherwise.

I'm especially keen to hear from:
- People running AI at scale who are bottlenecked by hardware
- Hardware folks who've been working on adaptive chips and have war stories
- Investors tracking the AI infrastructure layer
- Researchers who want to collaborate

No pitch, no deck. Just a conversation.

---

*If you got this far, thanks for reading. The full technical write-up, the source code, and the academic references are all linked in the [README](README.md) for the curious.*