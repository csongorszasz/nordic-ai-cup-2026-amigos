# Benchmark run

Policy: abd
Run ID: `198e59a769a7474cb8ecfdfab7168070`
Suite: quick; status: complete
Policy seed: fixed `1`
Cases: 3/3 recorded; 3 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 1112.2639554048662 |
| median | 1100.728669964696 |
| sample_std | 57.42131400084778 |
| min | 1061.485951131179 |
| max | 1174.5772451187236 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 1060.1000000001243 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 0.0 | 3 |
| Population: peak | 16.0 | 3 |
| Population: mean | 11.497650314645147 | 3 |
| Runtime: initialization_seconds | 1.6554009666666667 | 3 |
| Runtime: policy_construction_seconds | 0.0129175 | 3 |
| Runtime: simulation_seconds | 101.18744693333333 | 3 |
| Runtime: policy_seconds | 22.98560266666669 | 3 |
| Runtime: episode_seconds | 164.24783203333334 | 3 |
| Per-episode latency: batch_mean_ms | 2.1769721996860483 | 3 |
| Per-episode latency: batch_p50_ms | 2.1922 | 3 |
| Per-episode latency: batch_p95_ms | 4.269186666666667 | 3 |
| Per-episode latency: batch_max_ms | 18.037366666666667 | 3 |

## Limitations
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `aec603ad1a5651e2f20420ab3b86377f833ddb9fab7997bc652bc516f9794372`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
