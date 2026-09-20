# State of knowledge

Consolidated findings for the medical-appointment case, updated with the
2026-09-19 implementation audit.
Detail lives in `docs/tries.md` (experiments), `docs/adr/` (decisions),
`docs/next-steps.md` (plan) and `docs/local-serving.md` (runbook).

## The task and the score

One conversation (MP3) and ten yes/no questions per request; output ten booleans
plus an evidence span per yes. `score = 0.4·accuracy + 0.6·mean tIoU`, where
`mean tIoU` is averaged over **every annotated yes**, whether or not we answered
it. Evidence is the larger half, and a missed positive costs on both halves.

## Pipeline

Three answerers behind `MEDAPP_ANSWERER` (`answerers/`), with a never-raise
contract and shared ASR:

- `legacy` (default): `ASR → clause windows → proposition rewrite → NLI decision
  → guarded sub-range localization → answer + span`.
- `modernbert` (validated 0.606): `ASR → 64/32 passages → MiniLM top-8 retrieval
  → ModernBERT cross-encoder (3-way + token span + expected-tIoU) → answer +
  span`.
- `llm`: whole-transcript local instruction model, training-only demonstrations,
  and word-aligned evidence quotes. The observed active IDUN service is
  **26B/base + turbo int8**. The **0.744 official validation was E4B**, not
  a verified score for the current 26B process.

## Current local localization checkpoint

The qualified 26B/base + turbo int8 release with its fixed start +0.2 s
correction scores **0.8007415 locally**, accuracy **389/390**, mean tIoU
**0.6696119**. The provenance-bound source-event audit reproduced every
qualified response and that unrounded score. The demo-disjoint comparison
is a different cohort: **0.7978423** over 350 questions / 178 positives.

Source occurrence, evidence extent, and acoustic timing remain distinct
problems. A new hierarchy retaining short replies and multi-sentence episodes
has whole-unit oracle mIoU **0.882069**, falling to **0.838940** after the fixed
32-unit shortlist plus incumbent. These are diagnostic ceilings, not achieved
improvements; they do not support a near-perfect claim. The qualified service
is unchanged.

The complete pretrained inverse-question experiment did not improve source
selection: full composite scores were **0.5434** (context), **0.6000**
(source only), and **0.5804** (context ablation), versus **0.8007**.
Every demo-disjoint paired interval was negative. Reject these policies;
CPU feasibility and correct likelihood computation did not imply useful
reference localization.

Conditional CTC timing also failed as a direct replacement: **0.796676**
composite / **0.662837 mIoU** on all 390 questions, with 191 citations aligned
and three explicit unsupported-code fallbacks. The 172 previously eligible
citations were unchanged after repairing split-compound normalization.
All decisions stayed fixed. Its bounded local endpoint proposal oracle is
only **0.688711 mIoU**, not an achieved gain or a global dataset ceiling.
The next bounded check separates start/end clock contributions with grouped
training-only policy selection; it does not justify more alignment-model sweeps.

**Reference agreement is not always semantic correctness.** CSV references
`sample_63_yes_q02` and `sample_64_yes_q02` point to the opening greeting
instead of the later medical fact in both immutable, audio-matched full-v3 and
turbo transcripts. This is tracked in **#16** with **@Domynis** mentioned.
No labels or denominators were changed. Those two cases alone cannot explain
the plateau, but exact-span training should not mistake a high oracle score
on an unrelated greeting for successful medical grounding.

## What works

- **ASR.** `faster-whisper` with word timestamps; `large-v3` and `large-v3-turbo`
  preserve doses/numbers (`one million IU four times daily`,
  `fluconazole 50 milligrams`). Turbo is 2.2× faster than large-v3 on the 1650.
- **Learned answerer (ModernBERT).** A cross-encoder over retrieved passages
  with class-weighted training solves localization far better than NLI-scored
  sub-ranges: tIoU-when-yes **0.557 vs ~0.36**, validated **0.606** vs 0.545
  (T033/T034). Historical OOF numbers need rechecking after the provenance
  correction below. 64/32 passages + `multi-qa-MiniLM` top-8 retrieve the gold span
  for ~99.5 % of positives (T031).
- **Merged decision premise.** Deciding on clause ± 1 lifted positive recall
  **0.749 → 0.944** (large) at a tiny precision cost (ADR-0004), and validation
  confirmed the end-to-end gain (**0.457 → 0.545**, T027). It is a useful
  historical decision baseline; later LLM trials have substantially higher
  accuracy. The legacy/ModernBERT hybrid reported **0.647–0.661 OOF** (T035).
- **Hard negatives and off-topic.** 0.930 and 0.943 accuracy (large, merged) —
  NLI alone rejects the near-misses; the numeric guard is a no-op.
- **Proposition rewrite.** Decisive: accuracy `0.821` (base NLI, proposition)
  vs `0.692` for the raw question.
- **The endpoint.** Local + cloudflared: all 200 OK, worst 46.6 s of 60 s, zero
  timeouts on a 19-conversation dry run.

## What doesn't (and why)

**Localization includes both occurrence and boundary errors.** Earlier starts
alone do not prove a different occurrence. A read-only audit of the saved E4B
predictions found mIoU 0.5551, versus a gold-assisted within-quote trimming
oracle of 0.6865 and a quote +/- 24-word oracle of 0.8211. Corresponding 26B
values were 0.5935, 0.6950, and 0.8252. These are diagnostic ceilings, not
achieved scores. Choosing another exact match of the same quote recovered no
score on these runs. Different semantic restatements remain a separate issue.

**Measurement/provenance defects.** Cross-conversation ModernBERT negatives
previously grouped by passage source while retaining a different question's
source, allowing held-out questions into training. Training now splits before
example generation and tracks both sources. Prior probe results also preferred
large-v3 caches while deployment uses turbo. New comparisons must use explicit
transcripts, the serving response path, complete question coverage, and
demonstration-disjoint calibration.

**Failed refiners are evidence.** Three remote boundary-refiner trials reported
composed scores 0.6558, 0.6833, and 0.6732 versus E4B 0.7290. Their
gold-conditioned applicability gate was not reproducible at inference. Do not
repeat or deploy them unchanged.

Positive recall — previously the other failure — is resolved by the merged
premise (ADR-0004) and, for ModernBERT, by the hybrid decision (T035).

## Loss decomposition (T025, mIoU 0.340)

| stage | ceiling | loss | cause |
| --- | --- | --- | --- |
| any word range | 0.894 | — | ASR word times vs annotator boundaries |
| searched neighbourhood | 0.656 | −0.24 | most-entailing clause ≠ annotated clause |
| chosen span | 0.360 | −0.30 | NLI cannot rank by "what a human cites" |
| mean over positives | 0.340 | −0.02 | 11/195 positives answered no |

These headroom estimates describe T025, not the current LLM baseline.

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
| LLM L1 Gemma 4 E4B, in-sample (T038, not served) | 0.729 |
| **Gemma 4 E4B L1 served, validation (T039)** | **0.744** |
| LLM L1 Gemma 4 26B-A4B, in-sample (T040, not served) | 0.755 |

`dev_eval` predicted the service: training T024 (base-merged, 0.545) matched
validation T027 (0.545), ModernBERT OOF 0.610 matched validation T034 0.606,
and Gemma E4B in-sample 0.729 was close to validation T039 **0.744**.
These historical correspondences are encouraging, not proof of future
generalization. E4B remains the best documented officially validated method;
the active process is 26B. The continuing loop permits local validation only,
with no competition validation or evaluation submissions.

## Open problems / next

- **Comparable baseline:** full 26B/turbo replay, not a cached large-v3 proxy.
- **Grounding:** measure minimal-quote and segment-scoped prompts separately
  from boundary calibration; retain unchanged spans when refinement is uncertain.
- **Learning:** distinguish semantic support from preference for an annotated
  occurrence. Valid restatements must not be relabeled as false facts.
- **Audio:** pursue ASR/alignment changes only for measured residual loss.
- **Serving:** immutable IDUN runs, actual warm-up, owned-worker hard deadlines,
  bounded recovery, and unchanged public-entrypoint/rollback gates.
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
| isolated experiments | `idun/run.py`, `benchmark.py`, `results/idun_runs/` |
| boundary calibration | `calibrate_spans.py` (demonstration-disjoint grouped evaluation) |
| deadline isolation | `inference_worker.py` (opt-in, separate from the incumbent process) |
