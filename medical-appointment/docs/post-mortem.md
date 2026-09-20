# Post-mortem — Nordic AI Cup 2026, medical-appointment

**Final result:** official evaluation **0.779**, 6th place (Norway). Leader **0.808**.
**Submitted build:** `google/gemma-4-26b-a4b-it` @ `4d7ae498…`, `large-v3-turbo`
int8 ASR, base prompt, constant `+0.2 s` start boundary calibration, 1024 output
tokens, legacy special tokens, preloaded worker with warm-up.

Local HTTP acceptance was **0.8007** (accuracy 0.997, mean tIoU 0.6696); the
current validation set reproduced a deterministic **0.7701132841165913**. The
evaluation reproduced the historical best **0.779** exactly.

## 1. What we shipped

```
ASR (faster-whisper large-v3-turbo int8, word timestamps)
  -> whole-transcript LLM (26B-A4B, base prompt + 3 LOCO-safe demos, strict JSON)
  -> verbatim evidence_quote aligned back to ASR word times
  -> +0.2 s start calibration
```

Everything in `/predict` runs locally on one 80 GB GPU. The endpoint served
behind the stable ngrok domain `motor-throttle-viewer.ngrok-free.dev`, with a
GET-only watchdog. The calibration artifact is provenance-bound to the ASR
config hash (`e75a7f6e`), model revision, and `variant=base`.

## 2. The diagnosis: the loss is label/convention-limited

Accuracy was solved (389/390 = 0.9974). Every remaining point was tIoU, and the
evidence says that loss is the **annotation convention and a handful of bad
labels**, not model capability:

- Of the 195 positives, **22 are disjoint** (tIoU = 0). Classified against an
  independent agent draft (`annotations/drafts/ALL.json`):
  **8 genuine model errors, 11 subjective/convention** (the agent sides with the
  model, not gold), **3 degenerate gold** (e.g. `"Good morning," [0.00, 0.26]`).
- The independent agent's spans align with the **model** (tIoU 0.684) more than
  with **gold** (0.589); a consensus medoid scores 0.762 vs gold.
- Oracles: raw-word 0.927 mIoU, within-quote 0.751, quote ± 24 words 0.876.
  The timing representation could support far better spans than are scored.
- **Counterfactual:** crediting the 14 non-genuine cases (11 subjective +
  3 degenerate) raises mean tIoU 0.6696 → **0.7414** (score 0.8007 → **0.8438**).
  Excluding degenerate gold alone gives 0.6801.
- A **0.808** leader therefore sits *inside* the achievable ceiling: the winning
  delta is occurrence/boundary-convention recovery, not a different model.

Per-question detail, the disjoint-cause breakdown, and the genuine-error cases
are in `notebooks/tio_losses.ipynb`.

## 3. Experiment ledger (all rejected)

| family | trials | outcome |
|---|---|---|
| Prompt/format | v1, v2, v3, scoped, complete, audit, final_statement, no_timestamps, full_context, two_positive, reason, v1_reason, occurrence, multi3; AStra claim/answer-first | all ≤ base; best ≈ +0.004 in-sample, failed validation; `v1_reason` official **0.771** (raw) |
| Capacity/model | E4B 0.744, 26B 0.758→0.779, 31B dense tie (0.7978), Qwen3.8-27B 0.757, gpt-oss-120b **0.739** | capacity does not transfer |
| Boundary/calibration | 64-combo offset grid, fine offsets, duration trims, endpoint-source attribution | only constant **+0.2 start** helped |
| Localizers | legacy NLI 0.545, ModernBERT 0.606, hybrid 0.65, ridge/encoder/anchor/supervised refiners, soft-target tagger **−0.29** | encoder heads dead |
| ASR/alignment | large-v3, turbo, Medical-Whisper, WhisperX, CTC, native-segment fingerprinting | no downstream gain; **0 paired spans** matched |
| Selectors/rerankers | multi-candidate oracle < base, s1/s2/s3a, inverse-question Flan-T5 −0.25 | all negative |
| Metric RL | GRPO OOF 0.804 vs SFT 0.7986 vs baseline 0.7978, CI includes 0 | inconclusive |

AStra's parallel loop additionally ran a **native-unit selector** that regressed
(56 improved / 116 regressed / 23 tied; latency 18.8 s → 36.4 s) and was not
published. Its native-segment fingerprinting found **no reproducible unit**
across Whisper/WhisperX families.

## 4. Operational findings worth keeping

- **`+0.2` was verified applied.** A local replay of two captured validation
  conversations (`replay_parity.py`) matched the endpoint exactly when the
  calibration was on, and differed by exactly **+0.200** start when off.
- **Validation 0.770 vs evaluation 0.779 is set/version, not a bug.** The
  validation score is deterministic (bit-identical across runs); the evaluation
  dataset differs by design.
- **gpt-oss-120b** needed a run-local torch-2.8 overlay (`accelerate`, `kernels`,
  `triton≥3.4`) for MXFP4. With reasoning it is 54–89 s/conv and partly
  unparseable; a **no-reasoning chat-template workaround** (pre-close the
  analysis channel; `gptoss_no_reasoning.jinja`) drops it to 24–26 s with clean
  JSON but costs quality (0.739). Not servable as an improvement.
- **ngrok limits were non-binding.** After many validations the monthly data
  transfer was ~1.3 MB; the 1 GB cap, endpoint/agent counts, and rate limits
  never threatened the attempt.
- **One serving allocation, exclusive node.** A shared node caused a port-9054
  bind failure; `--exclusive` fixed it. Warm-up and an owned worker kept p95
  under budget.

## 5. Why we plateaued (mechanism)

`mean tIoU` is averaged over *every* annotated yes. The generator's chosen
occurrence/extent is (a) often one of several valid mentions, (b) inconsistent
in extent (neither clauses nor sentences), and (c) occasionally wrong. A
stronger model still produces a *valid* answer; it cannot guess the generator's
arbitrary pick. Every prompt change traded disjoint wins against boundary
losses; learned localizers overfit 195 noisy spans; capacity and ASR changes
never touched *which valid mention* is cited.

## 6. What the winner most likely did

Recovered the annotator's occurrence/extent convention better — the only axis
with headroom. Candidate mechanisms we did not reach: reproducing the
generator's own transcript units and snapping to them; a validated
occurrence-selection model; or metric-aware post-training with enough signal to
generalize. All sit within the 0.8438 counterfactual ceiling.

## 7. What we would do differently

1. **Attack the labels first.** Curate/repair degenerate and ambiguous gold
   spans; treat the generator as the imitation target, not semantic correctness.
2. **Run generator imitation early** (recover the source segmenter → learn unit
   selection → snap to the 20 ms grid), not in the last hours. It is the only
   principled route past 0.78.
3. **Metric-aware training sooner:** GRPO was directionally right but too late
   and too small; start it once the incumbent is frozen.
4. **Budget the one-shot attempt.** Keep the frozen endpoint untouched, confirm
   parity on validation, and queue the evaluation with margin.
5. **Operate like production:** exclusive node, watchdog, immutable runs, and
   provenance-bound artifacts (which we did do well).

## 8. Artifacts and commits

| artifact | location |
| --- | --- |
| diagnosis notebook (12 figures) | `notebooks/tio_losses.ipynb` |
| calibration artifact | `calibration/span_offset_base.json` |
| replay-parity diagnostic | `replay_parity.py`, `idun/job_replay_parity.slurm` |
| gpt-oss no-reasoning template | `gptoss_no_reasoning.jinja`, `answerers/llm_client.py` |
| prompt/subset experiments | `answerers/llm_prompt.py`, `compare_subset.py`, `idun/job_*` |
| knowledge | `docs/state-of-knowledge.md`, `docs/tries.md` |

Branch `medical-dominic-add-ngrok`: `9fc93fc`, `4a9b341`, `16efbfb` (AStra fixes),
`0137594` (prompt arms + subset diagnostic + DNS grace), `081bae9`/`59985a3`/
`6543178`/`e629b9e` (notebook + replay/gpt-oss probes + template). Nothing was
pushed, and the frozen endpoint and ngrok tunnel were shut down after the
evaluation.
