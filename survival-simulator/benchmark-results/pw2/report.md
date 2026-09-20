# Benchmark diagnostics - not a ranking

Policy: population-worker-pilot
Run ID: `16d202b688a049bab2b7ed909613ab30`
Suite: quick; status: truncated
Policy seed: fixed `1`
Cases: 3/3 recorded; 0 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 4.0849333333333355 |
| median | 4.099000000000002 |
| sample_std | 0.03377355967814655 |
| min | 4.046400000000002 |
| max | 4.109400000000002 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 4.000000000000002 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 5.0 | 3 |
| Population: peak | 5.0 | 3 |
| Population: mean | 5.0 | 3 |
| Runtime: initialization_seconds | 3.572658166666667 | 3 |
| Runtime: policy_construction_seconds | 0.03348906666666667 | 3 |
| Runtime: simulation_seconds | 0.305558 | 3 |
| Runtime: policy_seconds | 0.05094669999999999 | 3 |
| Runtime: episode_seconds | 4.121259133333333 | 3 |
| Per-episode latency: batch_mean_ms | 1.306325641025641 | 3 |
| Per-episode latency: batch_p50_ms | 1.2930666666666666 | 3 |
| Per-episode latency: batch_p95_ms | 1.8313100000000002 | 3 |
| Per-episode latency: batch_max_ms | 2.5002333333333335 | 3 |

## Limitations
- Diagnostic results only; observed subsets are not a complete policy ranking.
- A diagnostic step cap was configured, even if an episode ended before the cap.
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `44ddcf7189e1c9e804a00f0e26aef11de8592f010cd064a22ad3253a519ab0d9`
- Policy SHA-256: `adc1a76db2ef1c0c23fb932bae1b16fce3e9d42562cf2c9b4f37d8237c110679`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
