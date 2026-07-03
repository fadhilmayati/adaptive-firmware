# Adaptive Firmware Layer — Benchmark Leaderboard

_Generated from 40 benchmark runs_

## Overall ranking (by avg_reward)

| Rank | Workload | Agent | Avg Reward | Total Time (ms) | Total Energy (mJ) | Cache Hit |
|------|----------|-------|-----------:|----------------:|------------------:|----------:|
| 1 | audio_encoder | static_2 | 0.7639 | 5.04 | 1.21 | 83.3% |
| 2 | audio_encoder | tabular | 0.7514 | 13.04 | 2.82 | 66.7% |
| 3 | audio_encoder | static_2 | 0.7076 | 5.04 | 1.21 | 83.3% |
| 4 | cv_detection | static_2 | 0.6911 | 5.03 | 1.17 | 83.3% |
| 5 | cv_detection | tabular | 0.6911 | 5.03 | 1.17 | 83.3% |
| 6 | cv_detection | static_2 | 0.6458 | 5.03 | 1.17 | 83.3% |
| 7 | llm_prefill | tabular | 0.6387 | 36.05 | 7.32 | 81.2% |
| 8 | audio_encoder | profile | 0.5897 | 8.05 | 1.85 | 83.3% |
| 9 | audio_encoder | profile | 0.5897 | 8.05 | 1.85 | 83.3% |
| 10 | llm_prefill | static_2 | 0.5845 | 5.08 | 1.14 | 96.9% |
| 11 | llm_prefill | static_2 | 0.5626 | 5.08 | 1.14 | 96.9% |
| 12 | llm_prefill | profile | 0.5160 | 22.09 | 4.52 | 87.5% |
| 13 | llm_prefill | profile | 0.5160 | 22.09 | 4.52 | 87.5% |
| 14 | audio_encoder | static_3 | 0.5125 | 3.10 | 0.69 | 83.3% |
| 15 | cv_detection | static_3 | 0.5125 | 3.08 | 0.67 | 83.3% |
| 16 | llm_prefill | static_3 | 0.4942 | 3.12 | 0.66 | 96.9% |
| 17 | audio_encoder | tabular | 0.4809 | 24.05 | 5.00 | 33.3% |
| 18 | llm_prefill | tabular | 0.4790 | 77.06 | 15.52 | 59.4% |
| 19 | cv_detection | tabular | 0.4488 | 19.04 | 3.96 | 50.0% |
| 20 | cv_detection | profile | 0.4238 | 8.06 | 1.80 | 83.3% |
| 21 | cv_detection | profile | 0.4238 | 8.06 | 1.80 | 83.3% |
| 22 | audio_encoder | static_3 | 0.4000 | 3.10 | 0.69 | 83.3% |
| 23 | cv_detection | static_3 | 0.4000 | 3.08 | 0.67 | 83.3% |
| 24 | mixed_production | static_3 | 0.3835 | 35.51 | 11.09 | 100.0% |
| 25 | mixed_production | profile | 0.3794 | 537.32 | 111.76 | 99.0% |
| 26 | mixed_production | profile | 0.3793 | 528.34 | 109.98 | 99.0% |
| 27 | llm_prefill | static_3 | 0.3713 | 3.12 | 0.66 | 96.9% |
| 28 | llm_decode | static_3 | 0.3690 | 3.58 | 0.74 | 99.4% |
| 29 | mixed_production | tabular | 0.3639 | 636.43 | 142.80 | 98.7% |
| 30 | llm_decode | profile | 0.3414 | 30.56 | 6.15 | 96.9% |
| 31 | llm_decode | profile | 0.3410 | 33.55 | 6.75 | 96.2% |
| 32 | llm_decode | tabular | 0.3123 | 66.24 | 13.38 | 91.9% |
| 33 | mixed_production | tabular | 0.3086 | 751.41 | 165.81 | 98.4% |
| 34 | mixed_production | static_2 | 0.2986 | 25.77 | 19.88 | 100.0% |
| 35 | mixed_production | static_2 | 0.2719 | 25.77 | 19.88 | 100.0% |
| 36 | llm_decode | static_2 | 0.2671 | 5.39 | 1.21 | 99.4% |
| 37 | mixed_production | static_3 | 0.2394 | 35.51 | 11.09 | 100.0% |
| 38 | llm_decode | static_2 | 0.2353 | 5.39 | 1.21 | 99.4% |
| 39 | llm_decode | static_3 | 0.2231 | 3.58 | 0.74 | 99.4% |
| 40 | llm_decode | tabular | 0.2205 | 151.28 | 30.39 | 81.9% |

## Best agent per workload

| Workload | Best Agent | Avg Reward | Avg Time (ms) | Avg Energy (mJ) |
|----------|-----------|-----------:|--------------:|----------------:|
| audio_encoder | static_2 | 0.7639 | 5.04 | 1.21 |
| cv_detection | static_2 | 0.6911 | 5.03 | 1.17 |
| llm_decode | static_3 | 0.3690 | 3.58 | 0.74 |
| llm_prefill | tabular | 0.6387 | 36.05 | 7.32 |
| mixed_production | static_3 | 0.3835 | 35.51 | 11.09 |

## Head-to-head: adaptive vs static

For each workload, the adaptive agent's reward vs the best static config.

| Workload | Adaptive | Best Static | Delta | Winner |
|----------|---------:|-----------:|------:|--------|
| audio_encoder | 0.7514 (tabular) | 0.7639 (static_2) | -0.0125 | static |
| cv_detection | 0.6911 (tabular) | 0.6911 (static_2) | +0.0000 | static |
| llm_decode | 0.3123 (tabular) | 0.3690 (static_3) | -0.0567 | static |
| llm_prefill | 0.6387 (tabular) | 0.5845 (static_2) | +0.0542 | adaptive |
| mixed_production | 0.3639 (tabular) | 0.3835 (static_3) | -0.0196 | static |

---

To reproduce: `python -m benchmarks.runner`
To submit your own results: open a PR with your JSON files in `benchmarks/results/`