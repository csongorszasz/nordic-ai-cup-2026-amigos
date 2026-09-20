# Benchmark run

Policy: br
Run ID: `f0f25013818241f9b9519158af96dfef`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 102.79008727729769 |
| median | 39.79300000000029 |
| sample_std | 114.08171315948458 |
| min | 34.09800000000022 |
| max | 234.47926183189256 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 94.59999999999758 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 17.666666666666668 | 3 |
| Population: mean | 7.555580545650606 | 3 |
| Runtime: initialization_seconds | 1.6661583333333334 | 3 |
| Runtime: policy_construction_seconds | 0.008083833333333333 | 3 |
| Runtime: simulation_seconds | 7.960470966666662 | 3 |
| Runtime: policy_seconds | 0.6760707000000011 | 3 |
| Runtime: episode_seconds | 13.167299 | 3 |
| Per-episode latency: batch_mean_ms | 0.6012014600692122 | 3 |
| Per-episode latency: batch_p50_ms | 0.50195 | 3 |
| Per-episode latency: batch_p95_ms | 1.362 | 3 |
| Per-episode latency: batch_max_ms | 2.4463 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `44bac77b16eaa333f5e3e63dd6347df30275b31abbc182d811c85393282717cb`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
