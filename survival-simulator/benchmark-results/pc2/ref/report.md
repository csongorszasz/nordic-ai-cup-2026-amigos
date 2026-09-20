# Benchmark run

Policy: ref
Run ID: `5bd5ef7d579c4882bfdff254a13b357f`
Suite: standard; status: complete
Policy seed: fixed `1`
Cases: 20/20 recorded; 20 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 20 |
| mean | 641.4438603659114 |
| median | 606.4834585137265 |
| sample_std | 200.60107203018174 |
| min | 405.02153816779384 |
| max | 1061.5826337875371 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 623.8000000000717 | 20 |
| Time-limit completion fraction | 0.0 | 20 |
| Population: initial | 5.0 | 20 |
| Population: final | 0.0 | 20 |
| Population: peak | 12.7 | 20 |
| Population: mean | 7.31173907560634 | 20 |
| Runtime: initialization_seconds | 1.70578048 | 20 |
| Runtime: policy_construction_seconds | 0.0118449 | 20 |
| Runtime: simulation_seconds | 41.661946734999994 | 20 |
| Runtime: policy_seconds | 4.491885275000001 | 20 |
| Runtime: episode_seconds | 63.99184721 | 20 |
| Per-episode latency: batch_mean_ms | 0.7164256838867445 | 20 |
| Per-episode latency: batch_p50_ms | 0.65381 | 20 |
| Per-episode latency: batch_p95_ms | 1.3908587499999996 | 20 |
| Per-episode latency: batch_max_ms | 20.325754999999997 | 20 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `cc1e26ca86ab7cc6e50bc04c7719dcac8b41bb82b5912103d9c89bc4621d6900`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
