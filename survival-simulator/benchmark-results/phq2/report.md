# Benchmark run

Policy: hierarchical-baseline
Run ID: `152f3a5e5d844720b7507218ecd77406`
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
| Runtime: initialization_seconds | 2.4664815 | 3 |
| Runtime: policy_construction_seconds | 0.004483866666666667 | 3 |
| Runtime: simulation_seconds | 83.91903353333352 | 3 |
| Runtime: policy_seconds | 8.285007933333349 | 3 |
| Runtime: episode_seconds | 166.39565473333334 | 3 |
| Per-episode latency: batch_mean_ms | 1.3951275018503548 | 3 |
| Per-episode latency: batch_p50_ms | 1.2217833333333334 | 3 |
| Per-episode latency: batch_p95_ms | 2.874024999999999 | 3 |
| Per-episode latency: batch_max_ms | 8.6851 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `8475f4d695292954e7e6027926590d6f6f4a5ccb533b0a8c217b00388241fafd`
- Policy SHA-256: `7ce5226ecdfdb7ad7b77b128d691d345b02560d235775c6c14bc729ce49766a4`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
