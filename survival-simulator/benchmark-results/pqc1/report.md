# Paired benchmark comparison

Suite: quick; 3 world seeds, 3 cases per run; repeats: 1.
Reference: hierarchical-baseline (`reference`).

## Survival-first leaderboard - native scores are secondary

| Rank | Role | Policy | Mean | Median | Sample std | Min | Max | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | candidate-1 | population-v1 | 1130.5746960568601 | 1139.6602673574646 | 254.73151454503608 | 871.4219463477762 | 1380.6418744653397 | 3 |
| 2 | reference | hierarchical-baseline | 627.5148062389968 | 678.4577954000446 | 117.98203522742814 | 492.6204185880138 | 711.466204728932 | 3 |

| Policy | Completed worlds | Worlds | Lower-tail survival | Worst survival |
| --- | --- | --- | --- | --- |
| population-v1 | 0 | 3 | 815.4000000001168 | 815.4000000001168 |
| hierarchical-baseline | 0 | 3 | 468.400000000038 | 468.400000000038 |

## Absolute paired deltas - candidate minus reference

| Candidate | Mean delta | 95% interval | Wins | Ties | Losses | Independent worlds |
| --- | --- | --- | --- | --- | --- | --- |
| population-v1 (candidate-1) | 503.0598898178634 | [378.80152775976234, 669.1756697364077] (estimated) | 3 | 0 | 0 | 3 |

Repeat-level paired differences are averaged within each world seed before bootstrapping.
Intervals: percentile method, 10000 resamples, fixed PCG64 seed 20260917.
Wins/ties/losses count paired cases with absolute tie tolerance 1e-09.

## Survival, population and runtime

| Role | Policy | Measurement | Mean across recorded episodes | n |
| --- | --- | --- | --- | --- |
| reference | hierarchical-baseline | Survival (simulated seconds) | 599.3666666667344 | 3 |
| reference | hierarchical-baseline | Time-limit completion fraction | 0.0 | 3 |
| reference | hierarchical-baseline | Population: initial | 5.0 | 3 |
| reference | hierarchical-baseline | Population: final | 0.0 | 3 |
| reference | hierarchical-baseline | Population: peak | 13.333333333333334 | 3 |
| reference | hierarchical-baseline | Population: mean | 8.583411613122244 | 3 |
| reference | hierarchical-baseline | Runtime: initialization_seconds | 2.4664815 | 3 |
| reference | hierarchical-baseline | Runtime: policy_construction_seconds | 0.004483866666666667 | 3 |
| reference | hierarchical-baseline | Runtime: simulation_seconds | 83.91903353333352 | 3 |
| reference | hierarchical-baseline | Runtime: policy_seconds | 8.285007933333349 | 3 |
| reference | hierarchical-baseline | Runtime: episode_seconds | 166.39565473333334 | 3 |
| reference | hierarchical-baseline | Per-episode latency: batch_mean_ms | 1.3951275018503548 | 3 |
| reference | hierarchical-baseline | Per-episode latency: batch_p50_ms | 1.2217833333333334 | 3 |
| reference | hierarchical-baseline | Per-episode latency: batch_p95_ms | 2.874024999999999 | 3 |
| reference | hierarchical-baseline | Per-episode latency: batch_max_ms | 8.6851 | 3 |
| candidate-1 | population-v1 | Survival (simulated seconds) | 1071.2000000000423 | 3 |
| candidate-1 | population-v1 | Time-limit completion fraction | 0.0 | 3 |
| candidate-1 | population-v1 | Population: initial | 5.0 | 3 |
| candidate-1 | population-v1 | Population: final | 0.0 | 3 |
| candidate-1 | population-v1 | Population: peak | 16.0 | 3 |
| candidate-1 | population-v1 | Population: mean | 10.567505676419946 | 3 |
| candidate-1 | population-v1 | Runtime: initialization_seconds | 2.3645042000000003 | 3 |
| candidate-1 | population-v1 | Runtime: policy_construction_seconds | 0.0067667000000000005 | 3 |
| candidate-1 | population-v1 | Runtime: simulation_seconds | 216.28815676666588 | 3 |
| candidate-1 | population-v1 | Runtime: policy_seconds | 63.42924876666664 | 3 |
| candidate-1 | population-v1 | Runtime: episode_seconds | 501.8190538666667 | 3 |
| candidate-1 | population-v1 | Per-episode latency: batch_mean_ms | 6.167139162031224 | 3 |
| candidate-1 | population-v1 | Per-episode latency: batch_p50_ms | 5.688466666666667 | 3 |
| candidate-1 | population-v1 | Per-episode latency: batch_p95_ms | 13.94648 | 3 |
| candidate-1 | population-v1 | Per-episode latency: batch_max_ms | 70.65243333333333 | 3 |

## Limitations
- Rankings are descriptive; confidence intervals are not automatic significance claims.
- Quick-suite results are exploratory: only a small set of world seeds is sampled.
- Survival ordering precedes native score.

## Provenance and measurements
Original run manifests, source fingerprints, policy configurations, runtime/timing context, per-run metric summaries and aligned plotting data are embedded in `summary.json`.
The paired case export is `paired_cases.csv`. Failed or truncated runs are not admitted.
Latency summaries are per-episode batch measurements, not pooled latency percentiles.
