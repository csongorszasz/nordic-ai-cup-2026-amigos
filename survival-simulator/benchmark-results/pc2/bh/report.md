# Benchmark run

Policy: bh
Run ID: `428f048e6183470d98d16776646c9425`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 627.5148062389968 |
| median | 678.4577954000446 |
| sample_std | 117.98203522742814 |
| min | 492.6204185880138 |
| max | 711.466204728932 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 599.3666666667344 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 13.333333333333334 | 3 |
| Population: mean | 8.583411613122244 | 3 |
| Runtime: initialization_seconds | 2.8266002666666665 | 3 |
| Runtime: policy_construction_seconds | 0.02272863333333333 | 3 |
| Runtime: simulation_seconds | 93.81995663333335 | 3 |
| Runtime: policy_seconds | 9.855049333333303 | 3 |
| Runtime: episode_seconds | 138.78885823333334 | 3 |
| Per-episode latency: batch_mean_ms | 1.6519005195412966 | 3 |
| Per-episode latency: batch_p50_ms | 1.5104333333333333 | 3 |
| Per-episode latency: batch_p95_ms | 3.2399466666666665 | 3 |
| Per-episode latency: batch_max_ms | 32.887366666666665 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `1160f05f606d44c314072fd563a8f0ec0a5154cfa0e6ce30c24008b395e9ffe3`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
