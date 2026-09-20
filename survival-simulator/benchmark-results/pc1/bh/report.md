# Benchmark run

Policy: bh
Run ID: `8f7a1ae771534603a991be507c60ea34`
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
| Runtime: initialization_seconds | 1.9190899666666665 | 3 |
| Runtime: policy_construction_seconds | 0.0123988 | 3 |
| Runtime: simulation_seconds | 70.38724373333328 | 3 |
| Runtime: policy_seconds | 7.2769944666666655 | 3 |
| Runtime: episode_seconds | 104.33238656666667 | 3 |
| Per-episode latency: batch_mean_ms | 1.2168149624244637 | 3 |
| Per-episode latency: batch_p50_ms | 1.1077666666666666 | 3 |
| Per-episode latency: batch_p95_ms | 2.508058333333332 | 3 |
| Per-episode latency: batch_max_ms | 6.574366666666666 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `6391c06a0bb0dc315ce384c19416815bc0c7ff40a5448ce353e851ce7d089ab7`
- Policy SHA-256: `1160f05f606d44c314072fd563a8f0ec0a5154cfa0e6ce30c24008b395e9ffe3`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
