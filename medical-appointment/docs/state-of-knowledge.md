# State of knowledge

Consolidated findings for the medical-appointment case, as of 2026-09-18.
Detail lives in `docs/tries.md` (experiments), `docs/adr/` (decisions),
`docs/next-steps.md` (plan) and `docs/local-serving.md` (runbook).

## The task and the score

One conversation (MP3) and ten yes/no questions per request; output ten booleans
plus an evidence span per yes. `score = 0.4·accuracy + 0.6·mean tIoU`, where
`mean tIoU` is averaged over **every annotated yes**, whether or not we answered
it. Evidence is the larger half, and a missed positive costs on both halves.

## Pipeline

Two answerers behind `MEDAPP_ANSWERER` (`answerers/`), both with a never-raise
contract and shared ASR:

- `legacy` (default): `ASR → clause windows → proposition rewrite → NLI decision
  → guarded sub-range localization → answer + span`.
- `modernbert` (validated 0.606): `ASR → 64/32 passages → MiniLM top-8 retrieval
  → ModernBERT cross-encoder (3-way + token span + expected-tIoU) → answer +
  span`.

## What works

- **ASR.** `faster-whisper` with word timestamps; `large-v3` and `large-v3-turbo`
  preserve doses/numbers (`one million IU four times daily`,
  `fluconazole 50 milligrams`). Turbo is 2.2× faster than large-v3 on the 1650.
- **Learned answerer (ModernBERT).** A cross-encoder over retrieved passages
  with class-weighted training solves localization far better than NLI-scored
  sub-ranges: tIoU-when-yes **0.557 vs ~0.36**, validated **0.606** vs 0.545
  (T033/T034). 64/32 passages + `multi-qa-MiniLM` top-8 retrieve the gold span
  for ~99.5 % of positives (T031).
- **Merged decision premise.** Deciding on clause ± 1 lifted positive recall
  **0.749 → 0.944** (large) at a tiny precision cost (ADR-0004), and validation
  confirmed the end-to-end gain (**0.457 → 0.545**, T027). It remains the best
  *decision* signal: a hybrid using the legacy decision with ModernBERT spans is
  worth **0.647–0.661 OOF** (T035).
- **Hard negatives and off-topic.** 0.930 and 0.943 accuracy (large, merged) —
  NLI alone rejects the near-misses; the numeric guard is a no-op.
- **Proposition rewrite.** Decisive: accuracy `0.821` (base NLI, proposition)
  vs `0.692` for the raw question.
- **The endpoint.** Local + cloudflared: all 200 OK, worst 46.6 s of 60 s, zero
  timeouts on a 19-conversation dry run.

## What doesn't (and why)

**Localization.** The learned answerer lifts tIoU-when-yes to **0.557** (from
~0.36 for the NLI selector), but the remaining dominant error is **occurrence
selection**: 119/156 predicted starts are earlier than gold, i.e. the model
cites a different restatement of the same evidence. Span lengths match; it is not
a boundary-width problem. Gold/refute span containment in retrieved passages is
~0.99, so this is the scorer's objective, not retrieval. Score loss from
localization (~0.21 to a perfect contained span) still exceeds the recall loss
(~0.067).

Positive recall — previously the other failure — is resolved by the merged
premise (ADR-0004) and, for ModernBERT, by the hybrid decision (T035).

## Loss decomposition (T025, mIoU 0.340)

| stage | ceiling | loss | cause |
| --- | --- | --- | --- |
| any word range | 0.894 | — | ASR word times vs annotator boundaries |
| searched neighbourhood | 0.656 | −0.24 | most-entailing clause ≠ annotated clause |
| chosen span | 0.360 | −0.30 | NLI cannot rank by "what a human cites" |
| mean over positives | 0.340 | −0.02 | 11/195 positives answered no |

Headroom is now almost entirely **localization**. Accuracy headroom ≈ 0.02,
localization ≈ 0.33.

## Budgets (GTX 1650, int8)

- VRAM: 1575 MiB / 4096 MiB (turbo + base NLI). `large-v3` fp16 does not fit
  alongside NLI.
- Latency: ASR 16.1 s mean / 21.6 s worst; NLI 19.2 s (decision 10.5,
  localization 8.8). Validation dry run mean 30.1 s, worst 46.6 s.

## Validation correspondence

| config | score |
| --- | --- |
| shipped baseline (floor) | 0.200 |
| single-clause served, validation (T023) | 0.457 |
| merged decision served, validation (T027) | 0.545 |
| ModernBERT OOF, training (T033) | 0.610 |
| **ModernBERT served, validation (T034)** | **0.606** |
| hybrid legacy-decision + ModernBERT span, OOF (T035, not served) | 0.647–0.661 |
| LLM L1 Qwen2.5-7B, in-sample (T036, not served) | 0.665 |
| hybrid Qwen-decision + ModernBERT-span, in-sample (T037) | 0.705 |
| **LLM L1 Gemma 4 E4B, in-sample (T038, not served)** | **0.729** |

`dev_eval` predicted the service: training T024 (base-merged, 0.545) matched
validation T027 (0.545), and ModernBERT OOF 0.610 matched validation T034 0.606.
We can iterate locally with confidence. The hybrid (T035) is the next no-retrain
gain; an LLM ceiling probe is planned (Thread B).

## Open problems / next

- **Hybrid decision** (next): legacy NLI decision + ModernBERT span, OOF
  0.647–0.661 (T035), with selective NLI gating for latency. See
  `docs/next-steps.md` A1.
- **Occurrence-aware localization**: train cross-occurrence negatives so the
  span head cites the annotated occurrence (the ~0.09–0.21 remaining loss).
- **Thread B: LLM ceiling probe** — local instruction model (7–8B on IDUN) for
  decision + quote-cited localization; `answerers/llm.py` behind the factory.
- **Thread C: medical ASR** — benchmark `Na0s/Medical-Whisper-Large-v3` (and a
  turbo fine-tune on PriMock57) against the downstream score.
- **Serving** — local + **named tunnel** (needs a Cloudflare domain; quick
  tunnel meanwhile, ADR-0002). ModernBERT served behind `MEDAPP_ANSWERER`.
- **Hygiene** — captures are debug-only; never train on validation/evaluation.

## Artifact map

| what | where |
| --- | --- |
| legacy pipeline | `asr.py`, `windows.py`, `questions.py`, `guards.py`, `verifier/`, `answer.py` |
| answerer factory | `answerers/` (`base`, `legacy`, `modernbert`, `passages`, `minilm`, `modernbert_data`, `modernbert_model`, `span_utils`) |
| entrypoint | `example.py` (dispatches on `MEDAPP_ANSWERER`) |
| scoring / experiments | `dev_eval.py`, `train_modernbert.py`, `docs/tries.md` |
| serving | `local/`, `docs/local-serving.md`, `capture.py` |
| IDUN dev | `idun/`, `docs/azure-gpu-quota.md` (why not Azure) |
| decisions | `docs/adr/` |
| plan | `docs/next-steps.md` |
