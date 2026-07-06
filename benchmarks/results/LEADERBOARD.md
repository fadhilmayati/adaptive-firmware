# Adaptive Firmware Layer — Benchmark Leaderboard

_Generated from 72+ benchmark runs (9 agents × 5 workloads × up to 2 seeds)_

## Overall ranking (by avg_reward)

| Rank | Workload | Agent | Avg Reward | Total Time (ms) | Total Energy (mJ) | Cache Hit |
|------|----------|-------|-----------:|----------------:|------------------:|----------:|
| 1 | audio_encoder | static_2 | 0.7076 | 5.04 | 1.21 | 83.3% |
| 2 | audio_encoder | tabular | 0.7076 | 5.04 | 1.21 | 83.3% |
| 3 | audio_encoder | neural | 0.6795 | 13.04 | 2.82 | 66.7% |
| 4 | audio_encoder | oracle | 0.6795 | 13.04 | 2.82 | 66.7% |
| 5 | llm_prefill | oracle | 0.6720 | 14.04 | 2.92 | 93.8% |
| 6 | cv_detection | oracle | 0.6458 | 5.03 | 1.17 | 83.3% |
| 7 | cv_detection | static_2 | 0.6458 | 5.03 | 1.17 | 83.3% |
| 8 | cv_detection | tabular | 0.6458 | 5.03 | 1.17 | 83.3% |
| 9 | audio_encoder | tabular | 0.5951 | 16.05 | 3.40 | 50.0% |
| 10 | llm_prefill | neural | 0.5934 | 39.06 | 7.97 | 78.1% |
| 11 | audio_encoder | profile | 0.5897 | 8.05 | 1.85 | 83.3% |
| 12 | audio_encoder | smart_static | 0.5897 | 8.05 | 1.85 | 83.3% |
| 13 | cv_detection | tabular | 0.5797 | 11.03 | 2.37 | 66.7% |
| 14 | llm_prefill | static_2 | 0.5626 | 5.08 | 1.14 | 96.9% |
| 15 | cv_detection | neural | 0.5552 | 11.05 | 2.35 | 66.7% |
| 16 | llm_prefill | tabular | 0.5173 | 73.06 | 14.73 | 62.5% |
| 17 | llm_prefill | profile | 0.5160 | 22.09 | 4.52 | 87.5% |
| 18 | llm_prefill | smart_static | 0.5160 | 22.09 | 4.52 | 87.5% |
| 19 | audio_encoder | static_3 | 0.5125 | 3.10 | 0.69 | 83.3% |
| 20 | cv_detection | static_3 | 0.5125 | 3.08 | 0.67 | 83.3% |
| 21 | llm_prefill | ucb | 0.5062 | 69.06 | 14.17 | 62.5% |
| 22 | llm_prefill | static_3 | 0.4942 | 3.12 | 0.66 | 96.9% |
| 23 | audio_encoder | ucb | 0.4880 | 17.07 | 3.70 | 50.0% |
| 24 | llm_prefill | tabular | 0.4313 | 98.07 | 19.71 | 43.8% |
| 25 | cv_detection | profile | 0.4238 | 8.06 | 1.80 | 83.3% |
| 26 | cv_detection | smart_static | 0.4238 | 8.06 | 1.80 | 83.3% |
| 27 | mixed_production | tabular | 0.3907 | 709.73 | 151.17 | 98.6% |
| 28 | mixed_production | static_3 | 0.3835 | 35.51 | 11.09 | 100.0% |
| 29 | mixed_production | smart_static | 0.3831 | 54.47 | 15.02 | 100.0% |
| 30 | mixed_production | profile | 0.3796 | 496.38 | 103.49 | 99.1% |
| 31 | mixed_production | profile | 0.3787 | 642.30 | 132.78 | 98.8% |
| 32 | mixed_production | ucb | 0.3763 | 559.88 | 117.53 | 98.8% |
| 33 | llm_decode | static_3 | 0.3690 | 3.58 | 0.74 | 99.4% |
| 34 | mixed_production | oracle | 0.3668 | 287.12 | 72.99 | 99.5% |
| 35 | mixed_production | neural | 0.3631 | 709.41 | 157.29 | 98.5% |
| 36 | mixed_production | tabular | 0.3630 | 762.45 | 168.00 | 98.4% |
| 37 | llm_decode | smart_static | 0.3457 | 22.56 | 4.55 | 97.5% |
| 38 | llm_decode | profile | 0.3450 | 22.56 | 4.55 | 97.5% |
| 39 | llm_decode | profile | 0.3410 | 36.56 | 7.35 | 96.2% |
| 40 | llm_decode | oracle | 0.3354 | 6.19 | 1.39 | 99.4% |
| 41 | mixed_production | static_2 | 0.2986 | 25.77 | 19.88 | 100.0% |
| 42 | llm_decode | tabular | 0.2888 | 131.26 | 26.38 | 85.0% |
| 43 | llm_decode | neural | 0.2830 | 138.26 | 27.79 | 84.4% |
| 44 | llm_decode | tabular | 0.2818 | 147.25 | 29.59 | 83.1% |
| 45 | cv_detection | ucb | 0.2743 | 21.06 | 4.33 | 33.3% |
| 46 | llm_decode | static_2 | 0.2671 | 5.39 | 1.21 | 99.4% |
| 47 | llm_decode | ucb | 0.2568 | 154.26 | 30.99 | 81.9% |
| 48 | audio_encoder | random | 0.0000 | 24.07 | 4.96 | 16.7% |
| 49 | cv_detection | random | 0.0000 | 24.05 | 4.96 | 16.7% |
| 50 | llm_decode | random | 0.0000 | 384.50 | 76.99 | 56.2% |
| 51 | llm_prefill | random | 0.0000 | 84.10 | 16.92 | 53.1% |
| 52 | mixed_production | random | 0.0000 | 22270.47 | 4465.81 | 50.6% |

## Best agent per workload

| Workload | Best Agent | Avg Reward | Avg Time (ms) | Avg Energy (mJ) |
|----------|-----------|-----------:|--------------:|----------------:|
| audio_encoder | static_2 | 0.7076 | 5.04 | 1.21 |
| cv_detection | oracle | 0.6458 | 5.03 | 1.17 |
| llm_decode | static_3 | 0.3690 | 3.58 | 0.74 |
| llm_prefill | oracle | 0.6720 | 14.04 | 2.92 |
| mixed_production | tabular | 0.3907 | 709.73 | 151.17 |

## Head-to-head: adaptive vs static

For each workload, the adaptive agent's reward vs the best static config.

| Workload | Adaptive (Best) | Best Static | Delta | Winner |
|----------|---------:|-----------:|------:|--------|
| audio_encoder | 0.7076 (tabular) | 0.7076 (static_2) | +0.0000 | static |
| cv_detection | 0.6458 (tabular) | 0.6458 (static_2) | +0.0000 | static |
| llm_decode | 0.2888 (tabular) | 0.3690 (static_3) | -0.0802 | static |
| llm_prefill | 0.5934 (neural) | 0.5626 (static_2) | +0.0308 | adaptive |
| mixed_production | 0.3907 (tabular) | 0.3835 (static_3) | +0.0072 | adaptive |

## Thompson-Sampling UCB Agent (single-run results)

The new Thompson sampling bandit (UCBAgent) uses Beta/Beta posteriors with PRIOR_STRENGTH=4.

| Workload | Avg Reward | vs Best Static | vs Tabular Adaptive |
|----------|-----------:|---------------:|-------------------:|
| audio_encoder | 0.4880 | -0.2196 | -0.2196 |
| cv_detection | 0.2743 | -0.3715 | -0.3715 |
| llm_decode | 0.2568 | -0.1122 | -0.0320 |
| llm_prefill | 0.5062 | -0.0564 | -0.0111 |
| mixed_production | 0.3763 | -0.0072 | -0.0144 |

**Multi-seed analysis (mixed_production, 8 seeds):** UCB 0.3847 ± 0.0091 beats tabular 0.3789 ± 0.0160, confirming Thompson sampling's sharper convergence on the dominant memory_bound phase. Both agents trail static_3 (0.3866), indicating the oracle gap persists for learning agents on this workload.

---

To reproduce: `python -m benchmarks.runner`
To submit your own results: open a PR with your JSON files in `benchmarks/results/`
