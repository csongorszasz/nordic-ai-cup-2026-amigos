# Benchmark diagnostics - not a ranking

Policy: append-only-io-probe
Run ID: `680b670db476485a8c5d87f98fa14ba1`
Suite: quick; status: truncated
Policy seed: fixed `1`
Cases: 3/3 recorded; 0 successful; 0 missing.

| Native score statistic | Value |
| --- | --- |
| count | 3 |
| mean | 20.32280000000002 |
| median | 20.273200000000017 |
| sample_std | 0.10080952335965454 |
| min | 20.25640000000002 |
| max | 20.438800000000025 |

## Survival, population and runtime

| Measurement | Mean across recorded episodes | n |
| --- | --- | --- |
| Survival (simulated seconds) | 20.000000000000014 | 3 |
| Time-limit completion fraction | 0.0 | 3 |
| Population: initial | 5.0 | 3 |
| Population: final | 5.0 | 3 |
| Population: peak | 5.0 | 3 |
| Population: mean | 5.0 | 3 |
| Runtime: initialization_seconds | 4.1547012 | 3 |
| Runtime: policy_construction_seconds | 0.042495366666666666 | 3 |
| Runtime: simulation_seconds | 1.7318307999999998 | 3 |
| Runtime: policy_seconds | 0.31853336666666654 | 3 |
| Runtime: episode_seconds | 7.056702033333333 | 3 |
| Per-episode latency: batch_mean_ms | 1.6006701842546065 | 3 |
| Per-episode latency: batch_p50_ms | 1.5272666666666666 | 3 |
| Per-episode latency: batch_p95_ms | 2.30978 | 3 |
| Per-episode latency: batch_max_ms | 8.1002 | 3 |

## Limitations
- Diagnostic results only; observed subsets are not a complete policy ranking.
- A diagnostic step cap was configured, even if an episode ended before the cap.
- Latency summaries describe per-episode batch-call statistics, not pooled latency percentiles.

## Provenance
- Engine SHA-256: `8c7209c2b8c0f02b5e800c8010d2893a3f6934632845b9f36b5920f0e135fe3a`
- Runner SHA-256: `4ce110080b1aeec22677fd428dee9c9e64facd64bea112cff2687595761384a8`
- Policy SHA-256: `e5a3862f1c7fc6aca35ac02a43640f34438d7afa7d62f89542bf6790fa789518`
- Git revision: `af36a9c90678b3f7f801e0f479b04c35525232c8`; dirty: True
- Full source hashes, policy configuration, runtime and timing context are in `manifest.json`.
- All measurements and sample counts are in `summary.json` and `episodes.csv`.
