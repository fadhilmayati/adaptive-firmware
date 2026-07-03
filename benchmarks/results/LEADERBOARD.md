# Adaptive Firmware Layer — Benchmark Leaderboard

_Generated from 35 benchmark runs_

## Overall ranking (by avg_reward)

| Rank | Workload | Agent | Avg Reward | Total Time (ms) | Total Energy (mJ) | Cache Hit |
|------|----------|-------|-----------:|----------------:|------------------:|----------:|
| 1 | audio_encoder | static_2 | 0.7076 | 5.04 | 1.21 | 83.3% |
| 2 | audio_encoder | neural | 0.6795 | 13.04 | 2.82 | 66.7% |
| 3 | audio_encoder | oracle | 0.6795 | 13.04 | 2.82 | 66.7% |
| 4 | audio_encoder | tabular | 0.6795 | 13.04 | 2.82 | 66.7% |
| 5 | llm_prefill | oracle | 0.6720 | 14.04 | 2.92 | 93.8% |
| 6 | cv_detection | oracle | 0.6458 | 5.03 | 1.17 | 83.3% |
| 7 | cv_detection | static_2 | 0.6458 | 5.03 | 1.17 | 83.3% |
| 8 | audio_encoder | profile | 0.5897 | 8.05 | 1.85 | 83.3% |
| 9 | llm_prefill | static_2 | 0.5626 | 5.08 | 1.14 | 96.9% |
| 10 | llm_prefill | profile | 0.5160 | 22.09 | 4.52 | 87.5% |
| 11 | audio_encoder | static_3 | 0.5125 | 3.10 | 0.69 | 83.3% |
| 12 | cv_detection | static_3 | 0.5125 | 3.08 | 0.67 | 83.3% |
| 13 | cv_detection | neural | 0.4990 | 13.04 | 2.77 | 66.7% |
| 14 | llm_prefill | static_3 | 0.4942 | 3.12 | 0.66 | 96.9% |
| 15 | llm_prefill | neural | 0.4859 | 63.07 | 12.73 | 65.6% |
| 16 | llm_prefill | tabular | 0.4821 | 79.08 | 15.91 | 56.2% |
| 17 | cv_detection | tabular | 0.4602 | 16.04 | 3.36 | 50.0% |
| 18 | cv_detection | profile | 0.4238 | 8.06 | 1.80 | 83.3% |
| 19 | mixed_production | tabular | 0.3917 | 692.24 | 147.52 | 98.6% |
| 20 | mixed_production | static_3 | 0.3835 | 35.51 | 11.09 | 100.0% |
| 21 | mixed_production | profile | 0.3794 | 503.33 | 104.87 | 99.0% |
| 22 | llm_decode | static_3 | 0.3690 | 3.58 | 0.74 | 99.4% |
| 23 | mixed_production | oracle | 0.3668 | 287.12 | 72.99 | 99.5% |
| 24 | mixed_production | neural | 0.3632 | 738.36 | 163.21 | 98.5% |
| 25 | llm_decode | profile | 0.3434 | 30.56 | 6.15 | 96.9% |
| 26 | llm_decode | oracle | 0.3354 | 6.19 | 1.39 | 99.4% |
| 27 | llm_decode | tabular | 0.3005 | 95.25 | 19.19 | 88.8% |
| 28 | mixed_production | static_2 | 0.2986 | 25.77 | 19.88 | 100.0% |
| 29 | llm_decode | neural | 0.2817 | 133.27 | 26.79 | 85.0% |
| 30 | llm_decode | static_2 | 0.2671 | 5.39 | 1.21 | 99.4% |
| 31 | audio_encoder | random | 0.0000 | 24.07 | 4.96 | 16.7% |
| 32 | cv_detection | random | 0.0000 | 24.05 | 4.96 | 16.7% |
| 33 | llm_decode | random | 0.0000 | 384.50 | 76.99 | 56.2% |
| 34 | llm_prefill | random | 0.0000 | 84.10 | 16.92 | 53.1% |
| 35 | mixed_production | random | 0.0000 | 22270.47 | 4465.81 | 50.6% |

## Best agent per workload

| Workload | Best Agent | Avg Reward | Avg Time (ms) | Avg Energy (mJ) |
|----------|-----------|-----------:|--------------:|----------------:|
| audio_encoder | static_2 | 0.7076 | 5.04 | 1.21 |
| cv_detection | oracle | 0.6458 | 5.03 | 1.17 |
| llm_decode | static_3 | 0.3690 | 3.58 | 0.74 |
| llm_prefill | oracle | 0.6720 | 14.04 | 2.92 |
| mixed_production | tabular | 0.3917 | 692.24 | 147.52 |

## Head-to-head: adaptive vs static

For each workload, the adaptive agent's reward vs the best static config.

| Workload | Adaptive | Best Static | Delta | Winner |
|----------|---------:|-----------:|------:|--------|
| audio_encoder | 0.6795 (neural) | 0.7076 (static_2) | -0.0281 | static |
| cv_detection | 0.4990 (neural) | 0.6458 (static_2) | -0.1468 | static |
| llm_decode | 0.3005 (tabular) | 0.3690 (static_3) | -0.0685 | static |
| llm_prefill | 0.4859 (neural) | 0.5626 (static_2) | -0.0767 | static |
| mixed_production | 0.3917 (tabular) | 0.3835 (static_3) | +0.0082 | adaptive |

---

To reproduce: `python -m benchmarks.runner`
To submit your own results: open a PR with your JSON files in `benchmarks/results/`