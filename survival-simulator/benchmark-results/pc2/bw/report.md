# Benchmark run

Policy: bw
Run ID: `f03302c636384e3b8547720d43a5a79a`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 734.0274207059907 |
| median | 722.3944602971594 |
| sample_std | 78.01586557847263 |
| min | 662.4812422432352 |
| max | 817.2065595775775 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 723.6333333334293 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 14.0 | 3 |
| Population: mean | 7.419163867430001 | 3 |
| Runtime: initialization_seconds | 9.391840566666668 | 3 |
| Runtime: policy_construction_seconds | 0.019923833333333335 | 3 |
| Runtime: simulation_seconds | 60.42792006666659 | 3 |
| Runtime: policy_seconds | 5.6519755999999886 | 3 |
| Runtime: episode_seconds | 95.5022692 | 3 |
| Per-episode latency: batch_mean_ms | 0.7866510131971568 | 3 |
| Per-episode latency: batch_p50_ms | 0.6998833333333333 | 3 |
| Per-episode latency: batch_p95_ms | 1.5586099999999998 | 3 |
| Per-episode latency: batch_max_ms | 34.86946666666667 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `cc1e26ca86ab7cc6e50bc04c7719dcac8b41bb82b5912103d9c89bc4621d6900`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
