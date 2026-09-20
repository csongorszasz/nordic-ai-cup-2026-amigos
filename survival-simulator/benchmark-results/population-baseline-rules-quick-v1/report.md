# Benchmark diagnostics - not a ranking

Policy: aggressive-birth-baseline
Run ID: `8efa65e54d89487e9642ea6c491560b4`
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
| Runtime: initialization_seconds | 3.2533794 | 1 |
| Runtime: policy_construction_seconds | 0.0262314 | 1 |
| Runtime: simulation_seconds | 34.55623660000002 | 1 |
| Runtime: policy_seconds | 2.8292972999999986 | 1 |
| Runtime: episode_seconds | 75.4648934 | 1 |
| Per-episode latency: batch_mean_ms | 1.5219458310919851 | 1 |
| Per-episode latency: batch_p50_ms | 1.3242 | 1 |
| Per-episode latency: batch_p95_ms | 3.4533199999999997 | 1 |
| Per-episode latency: batch_max_ms | 8.001100000000001 | 1 |

## Limitations
- Diagnostic results only; observed subsets are not a complete policy ranking.
- No scored episodes are available; missing and failed scores are not zero.
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `9ad186899327c4bf413fc9599eb3222706d435112f2af06420edc962ea9538ce`
- Policy SHA-256: `296d9ef50bf2c1c028d6c0cb8be61dd86520dfe27510afbc9be722bcb8d98eac`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.

## Failure
telemetry: PermissionError: [WinError 5] Access is denied: 'benchmark-results\\population-baseline-rules-quick-v1\\telemetry\\world-1-repeat-0\\.latest.json.2bfdc74b36744c04933ca84e7ee143e4.pending' -> 'benchmark-results\\population-baseline-rules-quick-v1\\telemetry\\world-1-repeat-0\\latest.json'
