# Benchmark run

Policy: ab0
Run ID: `b8996ea7c7ca4a7a83a980f3a2cdc17b`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 998.0032355702568 |
| median | 944.4568606324422 |
| sample_std | 120.33965218612579 |
| min | 913.730391203074 |
| max | 1135.8224548752544 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 958.6333333334602 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 16.0 | 3 |
| Population: mean | 11.354640836679643 | 3 |
| Runtime: initialization_seconds | 2.3279755333333334 | 3 |
| Runtime: policy_construction_seconds | 0.0234777 | 3 |
| Runtime: simulation_seconds | 161.0171245000005 | 3 |
| Runtime: policy_seconds | 40.1125788000001 | 3 |
| Runtime: episode_seconds | 265.4642221666667 | 3 |
| Per-episode latency: batch_mean_ms | 4.11403708161588 | 3 |
| Per-episode latency: batch_p50_ms | 4.086266666666666 | 3 |
| Per-episode latency: batch_p95_ms | 8.318846666666667 | 3 |
| Per-episode latency: batch_max_ms | 34.164300000000004 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `69f1934f1a095cf5c61f57387aead36c5f67b0f49e4393a62c108bda4befd466`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
