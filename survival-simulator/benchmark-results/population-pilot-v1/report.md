# Benchmark diagnostics - not a ranking

Policy: src.policies.runtime:create_policy
Run ID: `0d77eed02c5141718f26f9aec323bde6`
Suite: quick; status: truncated
Policy seed: fixed `1`
Cases: 3/3 recorded; 0 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 20.457933333333354 |
| median | 20.451000000000022 |
| sample_std | 0.08221954349990919 |
| min | 20.37940000000002 |
| max | 20.543400000000023 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 20.000000000000014 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 6.333333333333333 | 3 |
| Population: peak | 6.666666666666667 | 3 |
| Population: mean | 6.036666666666666 | 3 |
| Runtime: initialization_seconds | 3.0998443666666664 | 3 |
| Runtime: policy_construction_seconds | 0.0196474 | 3 |
| Runtime: simulation_seconds | 1.7203554333333335 | 3 |
| Runtime: policy_seconds | 0.19311106666666666 | 3 |
| Runtime: episode_seconds | 9.982131033333333 | 3 |
| Per-episode latency: batch_mean_ms | 0.9704073701842546 | 3 |
| Per-episode latency: batch_p50_ms | 0.8744999999999999 | 3 |
| Per-episode latency: batch_p95_ms | 1.8411499999999998 | 3 |
| Per-episode latency: batch_max_ms | 3.0947333333333336 | 3 |

## Limitations
- Diagnostic results only; observed subsets are not a complete policy ranking.
- A diagnostic step cap was configured, even if an episode ended before the cap.
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `9ad186899327c4bf413fc9599eb3222706d435112f2af06420edc962ea9538ce`
- Policy SHA-256: `087dfffbbe6708f39cb95795641662dcc250dd56c0fb5e4e62b71e1313e6338f`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
