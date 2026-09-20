# Boundary calibration artifacts

Tracked copies of the small JSON artifacts read through `MEDAPP_SPAN_CALIBRATION`.
`models/` is gitignored and excluded from `idun/submit.sh` sync, so release
artifacts live here instead and travel with the branch.

## `span_offset_base.json`

- **Artifact sha256:** `740d73a80b5a12c53c23569d5da419952cca40d461f54ed21245e06cdbe59da6`
  (338 bytes).
- **Source run:** `offset-release-dns-2732ab44` on IDUN
  (`~/nordic-medical-runs/offset-release-dns-2732ab44/models/span_offset_base.json`),
  produced by the continuing improvement loop.
- **Method:** `constant-boundary-offsets`, start `+0.2 s`, end `0.0 s`,
  collapse-guarded (`answerers/boundaries.py:adjusted_span`).
- **Binding tuple:** `model=google/gemma-4-26b-a4b-it`,
  `revision=4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, `variant=base`,
  `asr_config_hash=e75a7f6e` (`WHISPER_MODEL=large-v3-turbo`,
  `WHISPER_COMPUTE_TYPE=int8`).
- **Fit provenance:** `source_records_sha256=dd9e708b1be158ffbea8f77c6818c7e81cc39edd42c6a7774e57522d274b0b9f`.
- **Selection:** `[0.2, 0.0]` chosen in every grouped 5-fold split,
  demonstration-disjoint; paired delta `+0.0121` (95% CI `0.0076–0.0170`).
- **Ceilings:** within-quote `0.7511`, quote±24 `0.8763`, raw-word `0.9273`.

`OffsetCalibration.validate_context` refuses the artifact if the serving model,
revision, variant or ASR config hash differs, so a stale binding fails at warmup
(`MEDAPP_REQUIRE_WARMUP=1`) rather than silently reshaping spans.
