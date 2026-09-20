# Benchmark diagnostics - not a ranking

Policy: wall-aware-baseline
Run ID: `18ceb9904f4e4c93ab09afce91afa7a9`
Suite: quick; status: failed
Policy seed: fixed `1`
Cases: 1/3 recorded; 0 successful; 2 missing.

| Native score statistic | Value |
| --- | --- |
| count | 0 |
| mean | unavailable |
| median | unavailable |
| sample_std | unavailable |
| min | unavailable |
| max | unavailable |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | unavailable | 0 |
| Time-limit completion fraction | unavailable | 0 |
| Population: initial | unavailable | 0 |
| Population: final | unavailable | 0 |
| Population: peak | unavailable | 0 |
| Population: mean | unavailable | 0 |
| Runtime: initialization_seconds | 3.2789911 | 1 |
| Runtime: policy_construction_seconds | 0.0210441 | 1 |
| Runtime: simulation_seconds | 75.82196529999995 | 1 |
| Runtime: policy_seconds | 7.514066300000012 | 1 |
| Runtime: episode_seconds | 154.4615741 | 1 |
| Per-episode latency: batch_mean_ms | 1.5244606005274903 | 1 |
| Per-episode latency: batch_p50_ms | 1.3481999999999998 | 1 |
| Per-episode latency: batch_p95_ms | 2.8078399999999983 | 1 |
| Per-episode latency: batch_max_ms | 7.830399999999999 | 1 |

## Limitations
- Diagnostic results only; observed subsets are not a complete policy ranking.
- No scored episodes are available; missing and failed scores are not zero.
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `9ad186899327c4bf413fc9599eb3222706d435112f2af06420edc962ea9538ce`
- Policy SHA-256: `d4f1eaa8d5a600b01cbd60911b1d148e026ffade250dab5ad88fc0305147fcf3`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.

## Failure
telemetry: PermissionError: [WinError 5] Access is denied: 'benchmark-results\\population-baseline-wall-aware-quick-v1\\telemetry\\world-1-repeat-0\\.latest.json.5f1994ee6fe74ef1b99888e8316522cd.pending' -> 'benchmark-results\\population-baseline-wall-aware-quick-v1\\telemetry\\world-1-repeat-0\\latest.json'
