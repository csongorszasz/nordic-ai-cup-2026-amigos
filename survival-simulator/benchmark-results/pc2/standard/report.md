# Benchmark run

Policy: standard
Run ID: `4609fc5e29a94ea6b1de391e5fbd8da3`
Suite: standard; status: complete
Policy seed: fixed `1`
Cases: 20/20 recorded; 20 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 20 |
| mean | 913.6692985547836 |
| median | 912.6057419518495 |
| sample_std | 191.20802176796062 |
| min | 531.4539148408453 |
| max | 1336.2114837991362 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 875.4200000001061 | 20 |
| Time-limit completion fraction | 0.0 | 20 |
| Population: initial | 5.0 | 20 |
| Population: final | 0.0 | 20 |
| Population: peak | 16.0 | 20 |
| Population: mean | 11.480491633387476 | 20 |
| Runtime: initialization_seconds | 1.9091620150000002 | 20 |
| Runtime: policy_construction_seconds | 0.017744455 | 20 |
| Runtime: simulation_seconds | 121.10933172999998 | 20 |
| Runtime: policy_seconds | 29.60037537500001 | 20 |
| Runtime: episode_seconds | 200.48444639 | 20 |
| Per-episode latency: batch_mean_ms | 3.3493968209737934 | 20 |
| Per-episode latency: batch_p50_ms | 2.9935025 | 20 |
| Per-episode latency: batch_p95_ms | 7.314758499999999 | 20 |
| Per-episode latency: batch_max_ms | 32.11797 | 20 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `faa83707a20d39ae65c9ebd285cd285c3416e8a611781c79db42d458d3c837a2`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
