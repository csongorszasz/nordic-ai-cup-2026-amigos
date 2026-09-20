# Benchmark run

Policy: p0
Run ID: `26ec0192caa24c3fa225af961d25c0f5`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 834.3555762463299 |
| median | 762.6283161433385 |
| sample_std | 155.49198790629515 |
| min | 727.6739207860284 |
| max | 1012.7644918096229 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 803.8333333334476 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 16.0 | 3 |
| Population: mean | 11.459729993741334 | 3 |
| Runtime: initialization_seconds | 2.1908702666666664 | 3 |
| Runtime: policy_construction_seconds | 0.016978366666666665 | 3 |
| Runtime: simulation_seconds | 136.29147466666672 | 3 |
| Runtime: policy_seconds | 34.23951416666663 | 3 |
| Runtime: episode_seconds | 227.07221679999998 | 3 |
| Per-episode latency: batch_mean_ms | 4.312725002903458 | 3 |
| Per-episode latency: batch_p50_ms | 3.971633333333333 | 3 |
| Per-episode latency: batch_p95_ms | 9.062738333333332 | 3 |
| Per-episode latency: batch_max_ms | 38.111666666666665 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `6391c06a0bb0dc315ce384c19416815bc0c7ff40a5448ce353e851ce7d089ab7`
- Policy SHA-256: `e5a3862f1c7fc6aca35ac02a43640f34438d7afa7d62f89542bf6790fa789518`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
