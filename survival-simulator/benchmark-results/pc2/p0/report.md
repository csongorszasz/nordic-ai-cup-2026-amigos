# Benchmark run

Policy: p0
Run ID: `65aed0b425c84aeab2a1d75a21837133`
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
| Runtime: initialization_seconds | 1.5886308 | 3 |
| Runtime: policy_construction_seconds | 0.015265266666666666 | 3 |
| Runtime: simulation_seconds | 86.66912383333333 | 3 |
| Runtime: policy_seconds | 22.600955466666676 | 3 |
| Runtime: episode_seconds | 144.94489520000002 | 3 |
| Per-episode latency: batch_mean_ms | 2.8530510526415456 | 3 |
| Per-episode latency: batch_p50_ms | 2.3694666666666664 | 3 |
| Per-episode latency: batch_p95_ms | 6.880749999999998 | 3 |
| Per-episode latency: batch_max_ms | 21.692866666666664 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `e5a3862f1c7fc6aca35ac02a43640f34438d7afa7d62f89542bf6790fa789518`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
