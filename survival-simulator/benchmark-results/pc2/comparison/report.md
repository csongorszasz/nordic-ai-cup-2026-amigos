# Paired benchmark comparison

Suite: standard; 20 world seeds, 20 cases per run; repeats: 1.
Reference: ref (`reference`).

## Survival-first leaderboard - native scores are secondary

| Rank | Role | Policy | Mean | Median | Sample std | Min | Max | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | candidate-1 | standard | 913.6692985547836 | 912.6057419518495 | 191.20802176796062 | 531.4539148408453 | 1336.2114837991362 | 20 |
| 2 | reference | ref | 641.4438603659114 | 606.4834585137265 | 200.60107203018174 | 405.02153816779384 | 1061.5826337875371 | 20 |

| Policy | Completed worlds | Worlds | Lower-tail survival | Worst survival |
| --- | --- | --- | --- | --- |
| standard | 0 | 20 | 508.4000000000471 | 508.4000000000471 |
| ref | 0 | 20 | 379.0000000000177 | 379.0000000000177 |

## Absolute paired deltas - candidate minus reference

| Candidate | Mean delta | 95% interval | Wins | Ties | Losses | Independent worlds |
| --- | --- | --- | --- | --- | --- | --- |
| standard (candidate-1) | 272.22543818887215 | [184.99813855354006, 353.6619523820891] (estimated) | 18 | 0 | 2 | 20 |

Repeat-level paired differences are averaged within each world seed before bootstrapping.
Intervals: percentile method, 10000 resamples, fixed PCG64 seed 20260917.
Wins/ties/losses count paired cases with absolute tie tolerance 1e-09.

## Survival, population and runtime

| Role | Policy | Measurement | Mean across recorded episodes | n |
| --- | --- | --- | --- | --- |
| reference | ref | Survival (simulated seconds) | 623.8000000000717 | 20 |
| reference | ref | Time-limit completion fraction | 0.0 | 20 |
| reference | ref | Population: initial | 5.0 | 20 |
| reference | ref | Population: final | 0.0 | 20 |
| reference | ref | Population: peak | 12.7 | 20 |
| reference | ref | Population: mean | 7.31173907560634 | 20 |
| reference | ref | Runtime: initialization_seconds | 1.70578048 | 20 |
| reference | ref | Runtime: policy_construction_seconds | 0.0118449 | 20 |
| reference | ref | Runtime: simulation_seconds | 41.661946734999994 | 20 |
| reference | ref | Runtime: policy_seconds | 4.491885275000001 | 20 |
| reference | ref | Runtime: episode_seconds | 63.99184721 | 20 |
| reference | ref | Per-episode latency: batch_mean_ms | 0.7164256838867445 | 20 |
| reference | ref | Per-episode latency: batch_p50_ms | 0.65381 | 20 |
| reference | ref | Per-episode latency: batch_p95_ms | 1.3908587499999996 | 20 |
| reference | ref | Per-episode latency: batch_max_ms | 20.325754999999997 | 20 |
| candidate-1 | standard | Survival (simulated seconds) | 875.4200000001061 | 20 |
| candidate-1 | standard | Time-limit completion fraction | 0.0 | 20 |
| candidate-1 | standard | Population: initial | 5.0 | 20 |
| candidate-1 | standard | Population: final | 0.0 | 20 |
| candidate-1 | standard | Population: peak | 16.0 | 20 |
| candidate-1 | standard | Population: mean | 11.480491633387476 | 20 |
| candidate-1 | standard | Runtime: initialization_seconds | 1.9091620150000002 | 20 |
| candidate-1 | standard | Runtime: policy_construction_seconds | 0.017744455 | 20 |
| candidate-1 | standard | Runtime: simulation_seconds | 121.10933172999998 | 20 |
| candidate-1 | standard | Runtime: policy_seconds | 29.60037537500001 | 20 |
| candidate-1 | standard | Runtime: episode_seconds | 200.48444639 | 20 |
| candidate-1 | standard | Per-episode latency: batch_mean_ms | 3.3493968209737934 | 20 |
| candidate-1 | standard | Per-episode latency: batch_p50_ms | 2.9935025 | 20 |
| candidate-1 | standard | Per-episode latency: batch_p95_ms | 7.314758499999999 | 20 |
| candidate-1 | standard | Per-episode latency: batch_max_ms | 32.11797 | 20 |

## Limitations
- Rankings are descriptive; confidence intervals are not automatic significance claims.
- Survival ordering precedes native score.

## Provenance and measurements
Original run manifests, source fingerprints, policy configurations, runtime/timing context, per-run metric summaries and aligned plotting data are embedded in `summary.json`.
The paired case export is `paired_cases.csv`. Failed or truncated runs are not admitted.
Latency summaries are per-episode batch measurements, not pooled latency percentiles.
