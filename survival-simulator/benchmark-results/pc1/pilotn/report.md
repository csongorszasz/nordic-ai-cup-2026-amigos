# Benchmark diagnostics - not a ranking

Policy: pilotn
Run ID: `aabf4367fed64b9cb3011dc8617491f3`
Suite: population-pilot-v1; status: truncated
Policy seed: fixed `1`
Cases: 4/4 recorded; 0 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 4 |
| mean | 6.127699999999995 |
| median | 6.140799999999994 |
| sample_std | 0.06314227321427891 |
| min | 6.046399999999995 |
| max | 6.182799999999995 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 5.999999999999995 | 4 |
| Time-limit completion fraction | 0.0 | 4 |
| Population: initial | 5.0 | 4 |
| Population: final | 5.0 | 4 |
| Population: peak | 5.0 | 4 |
| Population: mean | 5.0 | 4 |
| Runtime: initialization_seconds | 2.144791975 | 4 |
| Runtime: policy_construction_seconds | 0.017235825 | 4 |
| Runtime: simulation_seconds | 0.22664754999999998 | 4 |
| Runtime: policy_seconds | 0.043353225 | 4 |
| Runtime: episode_seconds | 2.553007125 | 4 |
| Per-episode latency: batch_mean_ms | 0.7348004237288135 | 4 |
| Per-episode latency: batch_p50_ms | 0.7021 | 4 |
| Per-episode latency: batch_p95_ms | 0.964632499999999 | 4 |
| Per-episode latency: batch_max_ms | 1.3481750000000001 | 4 |

## Limitations
- Diagnostic results only; observed subsets are not a complete policy ranking.
- A diagnostic step cap was configured, even if an episode ended before the cap.
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `6391c06a0bb0dc315ce384c19416815bc0c7ff40a5448ce353e851ce7d089ab7`
- Policy SHA-256: `e5a3862f1c7fc6aca35ac02a43640f34438d7afa7d62f89542bf6790fa789518`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
