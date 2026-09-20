# Benchmark run

Policy: abc
Run ID: `f1163e36d55d4e3bb4237a9ee047eb03`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 993.0742773032592 |
| median | 1065.369530732256 |
| sample_std | 205.45958419993568 |
| min | 761.2389197819003 |
| max | 1152.6143813956214 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 942.83333333345 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 16.0 | 3 |
| Population: mean | 12.866467711499993 | 3 |
| Runtime: initialization_seconds | 1.6966428999999998 | 3 |
| Runtime: policy_construction_seconds | 0.0148431 | 3 |
| Runtime: simulation_seconds | 101.3358872999997 | 3 |
| Runtime: policy_seconds | 26.624260833333295 | 3 |
| Runtime: episode_seconds | 172.75942396666667 | 3 |
| Per-episode latency: batch_mean_ms | 2.818147695205853 | 3 |
| Per-episode latency: batch_p50_ms | 2.7729 | 3 |
| Per-episode latency: batch_p95_ms | 5.022618333333333 | 3 |
| Per-episode latency: batch_max_ms | 34.34413333333333 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `02c54af66386c6d50df86390d433525dd56e3aa2ed468fdd5ac3ae797c70f5fc`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
