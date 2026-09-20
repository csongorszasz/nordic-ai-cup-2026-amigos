# Benchmark run

Policy: population-v1
Run ID: `a07230afd3594db8aa35651e51648714`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 1130.5746960568601 |
| median | 1139.6602673574646 |
| sample_std | 254.73151454503608 |
| min | 871.4219463477762 |
| max | 1380.6418744653397 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 1071.2000000000423 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 16.0 | 3 |
| Population: mean | 10.567505676419946 | 3 |
| Runtime: initialization_seconds | 2.3645042000000003 | 3 |
| Runtime: policy_construction_seconds | 0.0067667000000000005 | 3 |
| Runtime: simulation_seconds | 216.28815676666588 | 3 |
| Runtime: policy_seconds | 63.42924876666664 | 3 |
| Runtime: episode_seconds | 501.8190538666667 | 3 |
| Per-episode latency: batch_mean_ms | 6.167139162031224 | 3 |
| Per-episode latency: batch_p50_ms | 5.688466666666667 | 3 |
| Per-episode latency: batch_p95_ms | 13.94648 | 3 |
| Per-episode latency: batch_max_ms | 70.65243333333333 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `8475f4d695292954e7e6027926590d6f6f4a5ccb533b0a8c217b00388241fafd`
- Policy SHA-256: `d58c3137c39c6704ef41c0a1c5a4e82ee36aa141386c0ff4f430db63358719e7`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
