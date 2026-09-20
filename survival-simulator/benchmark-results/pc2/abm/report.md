# Benchmark run

Policy: abm
Run ID: `413c995cf1e44a5f8be1d93efec10587`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 502.9926602900091 |
| median | 376.30444473125186 |
| sample_std | 297.8117012912365 |
| min | 289.4714913464226 |
| max | 843.2020447923529 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 480.56666666670736 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 7.666666666666667 | 3 |
| Population: mean | 4.5232056044066855 | 3 |
| Runtime: initialization_seconds | 1.6521075333333333 | 3 |
| Runtime: policy_construction_seconds | 0.012535 | 3 |
| Runtime: simulation_seconds | 18.274637766666718 | 3 |
| Runtime: policy_seconds | 2.5129058333333374 | 3 |
| Runtime: episode_seconds | 27.6014287 | 3 |
| Per-episode latency: batch_mean_ms | 0.5721121396626588 | 3 |
| Per-episode latency: batch_p50_ms | 0.5414166666666667 | 3 |
| Per-episode latency: batch_p95_ms | 0.8873216666666665 | 3 |
| Per-episode latency: batch_max_ms | 21.48753333333333 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `45189d480a42e070e2c004cbe6c42598f0deb5a41d7d00e52a14f14e1d0ebb03`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
