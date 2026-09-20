# Benchmark run

Policy: abe
Run ID: `ad6c86ab17174c7e8a29017c7b98bc2c`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 1140.253052988239 |
| median | 1146.6008948253016 |
| sample_std | 50.774081951384666 |
| min | 1086.6035332519161 |
| max | 1187.5547308874995 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 1102.1666666667597 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 16.0 | 3 |
| Population: mean | 9.568391576584673 | 3 |
| Runtime: initialization_seconds | 1.6647084333333335 | 3 |
| Runtime: policy_construction_seconds | 0.0142272 | 3 |
| Runtime: simulation_seconds | 96.45215473333337 | 3 |
| Runtime: policy_seconds | 23.16441496666671 | 3 |
| Runtime: episode_seconds | 157.1518953 | 3 |
| Per-episode latency: batch_mean_ms | 2.096304003259998 | 3 |
| Per-episode latency: batch_p50_ms | 2.00055 | 3 |
| Per-episode latency: batch_p95_ms | 4.618200000000001 | 3 |
| Per-episode latency: batch_max_ms | 27.186766666666667 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `faa83707a20d39ae65c9ebd285cd285c3416e8a611781c79db42d458d3c837a2`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
