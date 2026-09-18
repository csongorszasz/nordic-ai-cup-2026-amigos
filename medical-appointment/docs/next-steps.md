# Next steps

Standing plan for the medical-appointment case, updated 2026-09-18 after the
served ModernBERT validation (**T034, 0.606**). Threads A/B/C below; execution
order is a judgement call, not a dependency.

## State

- **Validated score 0.606** (T034): `answerers/modernbert` behind
  `MEDAPP_ANSWERER`, final model on all 39, 64/32 passages + `multi-qa-MiniLM`
  top-8, ModernBERT-base cross-encoder, class weights, `psupport` decode τ 0.16.
  Previous best: legacy merged 0.545 (T027). Floor 0.200.
- **OOF predicts the service**: T033 OOF 0.610 vs T034 validation 0.606.
- The pipeline is now **modular** (`answerers/` behind `MEDAPP_ANSWERER`,
  default `legacy`), so legacy, ModernBERT and future answerers A/B against the
  same contract and the same harness.

## Diagnosis (T033 OOF, class-weighted ModernBERT)

| type | recall | failures |
| --- | --- | --- |
| positive | 0.800 (156/195) | 39 missed, `p_support` ≈ 0 (paraphrase/abbreviation) |
| hard_negative | 0.873 (124/142) | 18 false positives, all slot contradictions |
| off_topic | 1.000 | — |

- **Missed positives are paraphrases**, not threshold cases: `p_support` median
  0.002. Examples: "NSAID" ↔ "anti-inflammatory painkillers", "ECG" ↔
  "electrocardiogram", "proton pump inhibitor" ↔ "stomach protection".
  **35/39 are caught by the legacy large-NLI decision.**
- **False-positive hard negatives are slot contradictions** the model ignores:
  dose 200 vs 100 mg, LDL 4.2 vs 2.2, seven vs three lesions, "foreign body
  found" vs "no foreign body", antifungal vs moisturizing cream, throat vs mouth.
- **mean tIoU when answering yes = 0.557** (n=156); mIoU 0.446 = 156/195 × 0.557.
  tIoU buckets among yes: ≥0.8 → 40, 0.5–0.8 → 56, 0.3–0.5 → 25, 0.1–0.3 → 16,
  <0.1 → 19. Gold and predicted durations match (2.7 s vs 2.6 s), and **119/156
  predicted starts are earlier than gold** — the dominant error is picking a
  different occurrence of repeated/paraphrased evidence, not span length.
- Score arithmetic: recall loss ≈ 0.067; localization loss ≈ 0.21 (vs a perfect
  contained span) / ≈ 0.09 (vs a realistic 0.75).

## Improvement backlog (ranked)

| # | improvement | expected gain | effort / risk |
| --- | --- | --- | --- |
| 1 | **Hybrid: legacy NLI decision + ModernBERT span** (A: legacy decides; B: `MB_yes OR legacy_yes`) | OOF 0.610 → **0.647 (A) / 0.661 (B)**, no retrain | M / low |
| 2 | **Selective NLI gating** (run NLI only where MB says no / is uncertain) | keeps the hybrid at ~25–28 s mean, ~38 s worst | S / low |
| 3 | **Base vs large decision + `NLI_HALF=1` fp16** measurement | closes the gap to #1 or frees latency | S / low |
| 4 | **Occurrence-aware span training**: cross-occurrence negatives (same evidence restated elsewhere) | attacks the biggest loss, up to ~+0.09 | M / med |
| 5 | **Boundary head / frame-level refiner** around the chosen region | tIoU 0.557 → toward the 0.89 word-time ceiling | L / med |
| 6 | **Soft start/end targets + expected-tIoU loss** (currently near-inert) | modest mIoU; makes `q` usable | M / low |
| 7 | **Calibrate `p_support`** (temperature scaling on OOF) | principled `p > 0.4/(0.8+1.2q)` decoder, eval robustness | S / low |
| 8 | **Slot/contradiction features or head** (drug, dose, unit, polarity, anatomy) | hn 0.873 → legacy-like 0.93+ | M / med |
| 9 | **Capacity/ensembling**: ModernBERT-large, fold ensemble, LR schedule, longer training | a few points; fold spread 0.522–0.663 | M / med |
| 10 | **ASR/alignment bake-off** (medical Whisper, Parakeet, Qwen aligner) | small direct; enables #5 | L / med |
| 11 | **Retrieval fusion** (BM25 + MiniLM) / clause-anchor union | tiny — recall already 0.995/1.000 | S / low |
| 12 | **Data curation**: fix degenerate gold spans, review flagged rows, codify the occurrence convention | removes noise; feeds #4 | M / low |

Priority: **1 + 2 + 3** (safe, no retrain) → **4 + 7** → 5/8/9.

## Thread A — decide + localize

- **A1 (next): hybrid decision.** Add `answerers/hybrid.py` (factory name
  `hybrid`) that runs the legacy NLI *decision only* (merged clause premise +
  numeric guard, no sub-range search) and ModernBERT for spans, with a mode flag
  for strategy A vs B and selective NLI gating. Calibrate τ on grouped OOF.
  Gate: OOF > 0.610 and 1650 latency within budget.
- **A2: occurrence-aware span training.** Add cross-occurrence negatives to
  `modernbert_data.build_examples` (other mentions of the same evidence in the
  same conversation labelled NOT_MENTIONED / low target-tIoU) and re-train OOF.
- **A3: boundary refiner** (frame-level offsets) once A2 plateaus.
- **A4: calibration + decoder** — temperature-scale `p_support`, revisit the
  score-aware rule.
- **A5: capacity** — ModernBERT-large and/or fold ensembling.

## Thread B — LLM method (branch `medical-dominic-llm`)

**Done (T036/T038):** the probe is built and measured on the 39 conversations
(in-sample ceiling), transcript-only, plain `transformers` on V100-32G.

| model / rung | score | acc | mIoU | quote found |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B L0 | 0.588 | 0.962 | 0.339 | 0.776 |
| Qwen2.5-7B L1 | 0.665 | 0.982 | 0.454 | 0.954 |
| Qwen2.5-7B L2 | 0.611 | 0.985 | 0.362 | 0.855 |
| **Gemma 4 E4B L1** | **0.729** | **0.990** | **0.555** | **1.000** |

Few-shot is decisive; the decision is near-perfect (Gemma: positive 191/195,
hard_negative 142/142, off_topic 53/53). Gemma 4 E4B (small, modern,
instruction-tuned) beats the larger older Qwen2.5-7B by +0.064 and cuts the
worst localization cases (tIoU<0.1) from 47 to 22. Remaining loss is still
**occurrence selection** (22 cases; some are degenerate gold spans).

**Hybrid (T037):** Qwen decision + ModernBERT span scored **0.705** in-sample;
Gemma E4B alone (0.729) already exceeds it.

### Next
1. **Serving / servability**: E4B fp16 = 15.9 GB, int4 ~4 GB (borderline for the
   4 GB 1650); **E2B** int4 would fit — worth testing quality. Otherwise
   IDUN-hosted behind cloudflared (proven) with the job-lifetime/URL risk, or
   distill.
2. **Native audio (E4B/12B)**: the model accepts audio; probe whether it can
   ground evidence (and possibly bypass ASR) — separate experiment.
3. **Teacher (L4)**: use E4B to emit occurrence/paraphrase/slot labels to retrain
   ModernBERT's span head (Thread A).
4. **Occurrence selector**: feed top-k retrieved passages and ask which
   occurrence a human would cite; targets the remaining 22 cases.
5. **Risks**: in-sample ceiling (LOCO-safe few-shot), not validation; E4B is
   Apache 2.0. Confirm on validation before any flip.

## Thread C — medical-domain ASR

Unchanged from the earlier plan. `Na0s/Medical-Whisper-Large-v3` and a
PriMock57 turbo fine-tune benchmarked by downstream score, not WER. Pursue only
if it moves the score.

## Hygiene / open questions

- Captures (`captured/`) and transcripts are debug artifacts — never train on
  validation/evaluation data (evaluation is a different set).
- The validated 0.606 is behind `MEDAPP_ANSWERER=modernbert`; `legacy` remains
  the default and is one env var away.
- Keep `models/modernbert_final/final.pt`; the per-fold checkpoints are large and
  reproducible.
- Open: named tunnel vs quick tunnel for the one-shot evaluation; whether to
  fold the hybrid (#1) into the LLM branch or keep it on this branch.
