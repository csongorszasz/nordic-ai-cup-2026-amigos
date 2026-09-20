# Benchmark diagnostics - not a ranking

Policy: pilotn
Run ID: `bafb038a828d45a0b6dd28d7989b2469`
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
| Runtime: initialization_seconds | 3.1947336 | 4 |
| Runtime: policy_construction_seconds | 0.02944995 | 4 |
| Runtime: simulation_seconds | 0.40605025 | 4 |
| Runtime: policy_seconds | 0.074313975 | 4 |
| Runtime: episode_seconds | 3.9063279 | 4 |
| Per-episode latency: batch_mean_ms | 1.2595588983050847 | 4 |
| Per-episode latency: batch_p50_ms | 1.15025 | 4 |
| Per-episode latency: batch_p95_ms | 1.7639399999999996 | 4 |
| Per-episode latency: batch_max_ms | 2.40915 | 4 |

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
