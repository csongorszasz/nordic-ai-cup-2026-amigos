# Trials log

Append-only, newest last. The index is for scanning; the detail blocks below it
hold the config, the numbers and the reasoning. Machine-readable summaries come
from `dev_eval.py --json results/<tag>.json`; `results/` is gitignored and
excluded from `rsync --delete`, so pull it back with `bash idun/submit.sh pull`.

Metric conventions: `mIoU` is the official mean temporal IoU over the 195
annotated positives. Oracle rows are ceilings, not achieved scores. Unless a row
says otherwise, runs use `large-v3` ASR with cached transcripts (so ASR ≈ 0).

| # | date | change | model | acc | mIoU | score | notes |
|---|------|--------|-------|-----|------|-------|-------|
| T000 | 2026-09-17 | shipped baseline (always yes) | — | 0.500 | 0.000 | 0.200 | floor |
| T001 | 2026-09-17 | M1 plumbing + span diagnostics | — | — | — | — | pruned-candidate oracle 0.884 |
| T002 | 2026-09-17 | first verifier: proposition + NLI + clause span | base | 0.805 | 0.245 | 0.469 | positives weak (0.651) |
| T003 | 2026-09-17 | τ 0.5 → 0.3 | base | 0.821 | 0.270 | 0.490 | best of sweep A |
| T004 | 2026-09-17 | τ 0.2 | base | 0.808 | 0.276 | 0.488 | trades hn for pos |
| T005 | 2026-09-17 | raw question hypothesis | base | 0.692 | 0.162 | 0.374 | proposition is essential |
| T006 | 2026-09-17 | numeric guard off | base | 0.821 | 0.270 | 0.490 | identical to T003 — guard is a no-op |
| T007 | 2026-09-17 | instrument chosen vs neighbourhood | base | 0.821 | 0.270 | 0.490 | chosen 0.371 vs neigh 0.643 |
| T008 | 2026-09-17 | top-3 clauses, shortest_margin | base | 0.821 | 0.267 | 0.488 | plateau |
| T009 | 2026-09-17 | top-3, argmax_long | base | 0.821 | 0.262 | 0.485 | plateau |
| T010 | 2026-09-17 | top-3, argmax_short | base | 0.821 | 0.271 | 0.491 | plateau |
| T011 | 2026-09-17 | top-3, shortest_tau | base | 0.821 | 0.228 | 0.465 | shorter = worse |
| T012 | 2026-09-17 | top-2, shortest_margin | base | 0.821 | 0.267 | 0.488 | plateau |
| T013 | 2026-09-17 | greedy trim, margin 0.05 | base | 0.821 | 0.272 | 0.492 | trimming helps |
| T014 | 2026-09-17 | greedy trim, margin 0.1 | base | 0.821 | 0.277 | 0.494 | |
| T015 | 2026-09-17 | greedy trim, margin 0.2 | base | 0.821 | 0.259 | 0.484 | too loose |
| T016 | 2026-09-17 | greedy trim + top-3 | base | 0.821 | 0.293 | 0.504 | best base |
| T017 | 2026-09-17 | τ 0.4 + trim | base | 0.810 | 0.268 | 0.485 | fewer positives |
| T018 | 2026-09-17 | **large NLI**, τ 0.3, top-3, trim | large | 0.851 | 0.286 | **0.512** | best overall |
| T019 | 2026-09-17 | large NLI, τ 0.5 | large | 0.836 | 0.272 | 0.498 | |
| T020 | 2026-09-17 | large NLI, top-1 | large | 0.851 | 0.272 | 0.504 | |
| T023 | 2026-09-17 | **validation dry run** (served config) | base | — | — | **0.457** | local+cloudflared, all 200 OK |
| T024 | 2026-09-17 | **merged decision premise** (clause±1) | base | 0.882 | 0.320 | 0.545 | positive recall 0.892 |
| T025 | 2026-09-17 | merged decision, large NLI | large | **0.938** | **0.340** | **0.579** | positive recall 0.944 |
| T026 | 2026-09-17 | merged decision, large, τ 0.5 | large | 0.938 | 0.337 | 0.578 | |
| T027 | 2026-09-17 | **validation: served merged config** | base | — | — | **0.545** | matches T024 exactly |
| T028 | 2026-09-18 | merged + served caps + length-diverse cap | base | 0.882 | 0.314 | 0.541 | served-cap oracle 0.843 |
| T029 | 2026-09-18 | same, large NLI | large | 0.938 | 0.318 | 0.566 | caps cost 0.013 vs T025 |
| T030 | 2026-09-18 | OOF calibration run (τ=0) | large | — | — | 0.569 | LOCO τ 0.65 |
| T031 | 2026-09-18 | ModernBERT passage retrieval recall gate | MiniLM | — | — | — | 64/32 + multi-qa top-8: sup 0.995 / ref 1.000 |
| T032 | 2026-09-18 | ModernBERT scorer, first OOF (4 ep) | MB-base | 0.777 | 0.391 | 0.546 | +psupport decode 0.602 (LOCO τ .08) |
| T033 | 2026-09-18 | **ModernBERT scorer, class-weighted (6 ep)** | MB-base | 0.854 | 0.446 | **0.609** | LOCO τ .16 → 0.610 |
| T034 | 2026-09-18 | **validation: served ModernBERT** | MB-base | — | — | **0.606** | 19 convs, all 200 OK, worst 32.4 s |
| T035 | 2026-09-18 | hybrid OOF analysis (legacy decide + MB span) | large+MB | 0.933 | 0.456 | 0.647 | B (OR) 0.661; offline only |
| T036 | 2026-09-18 | **LLM probe L0/L1/L2** (39 convs, in-sample) | Qwen2.5-7B | 0.982 | 0.454 | **0.665** | L1 few-shot; L0 0.588, L2 0.611 |
| T037 | 2026-09-18 | hybrid sim: LLM decision + MB span | Qwen+MB | 0.982 | 0.521 | **0.705** | offline join of T036/T033 |
| T038 | 2026-09-18 | **LLM L1 with Gemma 4 E4B** (39 convs, in-sample) | Gemma4-E4B | 0.990 | 0.555 | **0.729** | quote_found 1.000, 0 parse fails |
| T039 | 2026-09-18 | **validation: served Gemma 4 E4B L1** | Gemma4-E4B | — | — | **0.744** | 19 convs, 14–19 s, IDUN + cloudflared |
| T040 | 2026-09-18 | LLM L1 Gemma 4 26B-A4B (39 convs, in-sample) | Gemma4-26B | 0.997 | 0.594 | **0.755** | 1 decision miss; 22 worst unchanged |

Models: `base` = `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`,
`large` = `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.

> Caveat: the T002–T017 `results/*.json` were deleted by an `rsync --delete` bug
> (results/ was not excluded); those metrics are transcribed from the job logs.
> Fixed — `results/` is now excluded and `submit.sh pull` retrieves it.

---

## T001 — M1 plumbing and span-oracle diagnostics

- **Hypothesis:** clause windows overlap every gold span; the ceiling depends on
  how tight the returned span can be.
- **Config:** stub answerer, diagnostics only.
- **Results:** 1654 windows (42.4/conversation); gold spans overlapping a window
  100%, fully contained 28.7%. Oracles: clause window 0.629, pruned candidate
  0.884, word range 0.894, padded window 0.599.
- **Conclusion:** window spans would cap tIoU at 0.63; the returned span must be
  a word sub-range (ADR-0001). Padding hurts.

## T002 — first real verifier

- **Config:** proposition hypothesis, base NLI, τ 0.5, clause-level decision,
  clause-neighbourhood sub-range search.
- **Results:** acc 0.805 (pos 0.651, hn 0.965, ot 0.943), mIoU 0.245, score
  0.469; 68 positives missed; tIoU-when-yes 0.377.
- **Conclusion:** too many positives answered no, and spans loose. Tune τ and the
  selector.

## T003–T006 — sweep A (decision)

- **T003** τ 0.3 → 0.821/0.270/0.490. **T004** τ 0.2 → 0.808/0.276/0.488
  (more positives, more hard-negative errors). **T005** raw question hypothesis →
  0.692/0.162/0.374: the proposition rewrite is doing the heavy lifting.
  **T006** guard off → bit-identical to T003.
- **Conclusion:** proposition > raw decisively; numeric guard fires on nothing at
  τ 0.3; τ 0.3 is the best balance.

## T007 — instrumentation

- Added `return_info` (winning clause, searched neighbourhood, chosen range).
- **Results:** for the 142 positives answered yes, chosen tIoU **0.371** while the
  best span in the *searched* neighbourhood is **0.643**. Decision paths:
  `below_threshold` 231, `candidate` 157.
- **Conclusion:** two gaps — the searched neighbourhood (0.884 → 0.643, wrong
  clause) and the selector (0.643 → 0.371).

## T008–T012 — sweep B (search breadth + selection rule)

- top-1/2/3 clauses × {shortest_margin, argmax_long, argmax_short, shortest_tau}.
- **Results:** all 0.821 accuracy, mIoU 0.228–0.271, score 0.465–0.491.
- **Conclusion:** neither breadth nor selection rule moves the needle. The
  objective is the problem, not the tunable.

## T013–T017 — sweep C (greedy trimming)

- Start from the highest-scoring candidate, peel words off both ends while
  entailment holds (margin γ).
- **Results:** γ 0.1 + top-3 (**T016**) → 0.821/0.293/0.504. Larger γ trims too
  far; τ 0.4 loses positives.
- **Conclusion:** trimming is a genuine improvement (spans were systematically the
  max 14–16 words), but still far below the 0.715 neighbourhood oracle.

## T018–T020 — sweep D (large NLI)

- Swapped the verifier to DeBERTa-v3-large.
- **Results:** **T018** acc **0.851** (pos 0.749, hn 0.944, ot 0.981), mIoU 0.286,
  score **0.512**. τ 0.5 (T019) trades positives; top-1 (T020) matches accuracy
  but loses mIoU.
- **Conclusion:** decision quality rises with NLI capacity; localization does not
  (chosen 0.381 vs neighbourhood 0.714 at T018).

---

## Findings

**Decision is a genuine NLI task; localization is not.** Accuracy tracks model
capacity (base 0.821 → large 0.851). Localization does not, because NLI scores
*truth under the whole premise*, which is monotone in context, while the gold
span is the *minimal* passage a human cites. The two objectives are
anti-correlated — the model's best-scoring span is the longest one.

**The tIoU loss decomposes as** (T018: mIoU 0.286):

| stage | ceiling | loss | cause |
|---|---|---|---|
| any word range | 0.894 | — | ASR words don't match annotator boundaries exactly |
| searched neighbourhood (top-3) | 0.714 | −0.18 | the most-entailing clause isn't the annotated one |
| chosen span | 0.381 | −0.33 | NLI can't rank by "what a human would cite" |
| mean over all positives | 0.286 | −0.10 | 49/195 positives answered no (each scores 0) |

**Score arithmetic:** `0.4·acc + 0.6·mIoU`. Accuracy headroom ≈ `0.4·0.15 = 0.06`;
localization headroom ≈ `0.6·(0.88 − 0.29) = 0.35`. Localization is where the
competition is.

**Numeric guard is currently dead weight** (T003 ≡ T006): it never changes a
decision at τ 0.3, because the clause containing the evidence also contains the
question's number. Keep it (cheap insurance) but don't count on it.

## T024–T026 — the recall bottleneck was premise construction

- **Diagnostic (`diagnose_recall.py`):** for 195 positives, entailment of the
  proposition against best clause / clause±1 merged / gold text. Merged lifted
  the ≥τ rate from **73% → 90%** (base) and **75% → 95%** (large); hard-negative
  false positives rose only 9% → 13% (base) and 5% → 7% (large). Gold-text scores
  were *lower* than merged (62/66%) because annotations are minimal, often
  truncated fragments.
- **Root cause:** the gold evidence crosses our clause boundaries **71%** of the
  time, so a single-clause premise could not entail it.
- **Fix:** decision premise = clause ± 1 merged (`MEDAPP_DECISION_NEIGHBOURS=1`).
- **Result:** positive recall **0.749 → 0.944**, accuracy **0.851 → 0.938**,
  mIoU **0.286 → 0.340**, score **0.512 → 0.579** (T025). Negligible latency cost.
- **Conclusion:** recall was premise construction, not model capability. The
  remaining gap is localization (chosen 0.36 vs neighbourhood oracle 0.66,
  global 0.88) — the M3 problem.

## T027 — served validation confirms the merged premise (0.545)

- **Config:** local GTX 1650 + cloudflared quick tunnel; `large-v3-turbo` int8 +
  base NLI; `DECISION_NEIGHBOURS=1`, `TOP_CLAUSES=1`, `greedy_trim`,
  `MAX_CANDIDATES=120`, `MAX_RANGE_WORDS=10`, τ 0.3, `MEDAPP_CAPTURE=1`.
- **Results:** 19 validation conversations, **all 200 OK, zero timeouts**;
  latency mean **21.6 s**, worst **31.3 s**; **service score 0.545**.
- **Recall:** validation yes-rate **37% → 47%** (near the balanced ~50%).
- **Correspondence:** 0.545 equals the training estimate T024 (base-merged,
  0.545) exactly — `dev_eval` is a faithful predictor of the service.
- **Conclusion:** ADR-0004 validated end-to-end (+0.088 over the single-clause
  dry run). The remaining headroom is localization (M3); large NLI would add
  ~0.03 but needs latency work on the 1650.

## T028–T030 — Tier 0 hardening verification

- **T028/T029 (candidate-cap fix):** the cap is now length-diverse (round-robin
  across span lengths) instead of truncating by shortest. Served-cap oracle
  **0.843** (vs 0.884 uncapped). Score: base **0.541**, large **0.566** — the
  served caps cost ~0.013 vs the uncapped T025 (0.579), which is the price of
  latency safety and is removed by the learned localizer.
- **T030 (OOF calibration):** `dev_eval --oof` with τ=0 emits per-question `p`
  and span for all 390 questions. `calibrate.py` (LOCO) selects **τ=0.65**
  (stable across folds) giving **0.569** (acc 0.949, mIoU 0.316). The
  score-aware formula gives τ≈0.331, but that assumes calibrated `p`; the
  empirical LOCO optimum is more reliable here.
- **Tier 0 (serving hardening):** audio-content cache keys, ASR compute-type
  fallback chain (fp16→int8→fp32, for P100), deadline fallback, warm-up
  inference, functional Dockerfile, `local/serve.sh` supervisor. Tests: 54.

## Latency (P100, transcripts cached, per conversation)

| config | per conversation |
|---|---|
| base, top-1, no trim (T002) | ~2.8 s |
| base, top-1, no trim (T003–T006) | ~2.1 s |
| base, greedy trim (T013–T017) | ~3.1 s |
| **large, top-3, trim (T018–T020)** | **~8.0 s** |

Drivers: clause scoring (~42×10 pairs), candidate scoring (up to 400×10),
greedy trim (sequential, unbatched), plus model load per run. Large NLI is
~2.5–3× base per pair. With ASR added, a GPU host stays inside the 60 s budget;
CPU would not.

## T023 — validation dry run (local serving)

- **Goal:** confirm the endpoint contract end-to-end and capture what the
  validation service sends.
- **Host:** local GTX 1650 (WSL) + cloudflared quick tunnel (ADR-0002).
- **Config:** `large-v3-turbo` int8 ASR + base NLI; `TOP_CLAUSES=1`,
  `greedy_trim`, `MAX_CANDIDATES=120`, `MAX_RANGE_WORDS=10`, τ 0.3,
  `MEDAPP_CAPTURE=1`.
- **Results:** 22 `/predict` calls (2 local tests + Verify + 19 validation),
  **all 200 OK, zero timeouts**; latency mean **30.1 s**, worst **46.6 s**;
  VRAM 1575 MiB. Service raw score **0.457**.
- **Answer behaviour:** yes on **37%** of questions (70/190) on a set that is
  exactly balanced (~50%) — the same positive under-call as the training trials.
- **Conclusion:** the local harness predicts the service (3-conv local config
  scored 0.471 vs 0.457 on validation). Captures saved to `captured/` (debug
  only). Local + tunnel is a viable serving path.

### Local ASR benchmark (GTX 1650, int8, 3 clips ~95 s mean)

| model | RTF | 3 clips | worst clip | VRAM | load |
| --- | --- | --- | --- | --- | --- |
| large-v3 | 3.6× | 77.8 s | 31.9 s | 1957 MiB | 15.9 s |
| **large-v3-turbo** | **7.9×** | 35.6 s | 13.8 s | 1125 MiB | 7.8 s |
| distil-large-v3 | 8.8× | 32.2 s | 12.4 s | 1061 MiB | 4.4 s |
| medium.en | 4.6× | 61.3 s | 25.2 s | 1029 MiB | 4.7 s |
| small.en | 7.6× | 37.2 s | 14.3 s | 389 MiB | 1.4 s |

Chosen serving config: ASR 16.1 s mean / 21.6 s worst + NLI 19.2 s
(decision 10.5, localization 8.8) ⇒ ~35 s mean, ~41 s worst per conversation.

## T031 — ModernBERT passage retrieval recall gate

- **Goal:** before training, confirm the sliding-window passages + MiniLM
  retrieval surface the annotated evidence. A gold span not present in the
  retrieved candidates cannot be recovered by any scorer.
- **Tooling:** `answerers/passages.py` (sliding windows from ASR words),
  `answerers/minilm.py` (MiniLM mean-pooled embeddings), and the gate in
  `answerers/modernbert_data.py` (`python -m answerers.modernbert_data --sweep`).
- **Finding:** the planned 28-word / 14-stride window leaves **5/195 gold spans
  unserveable by construction** (0.974 containment). 48/24 gives **195/195**.
- **Retriever:** `multi-qa-MiniLM-L6-cos-v1` beat the general `all-MiniLM`
  models on question→passage retrieval. At 48/24:

  | window | retriever | top-3 sup | top-5 sup | top-5 ref | top-8 sup | top-8 ref |
  | --- | --- | --- | --- | --- | --- | --- |
  | 28/14 | multi-qa-MiniLM-L6 | 0.831 | 0.903 | 0.840 | 0.938 | 0.920 |
  | 48/24 | multi-qa-MiniLM-L6 | 0.897 | 0.944 | 0.952 | 0.985 | 0.976 |
  | 64/32 | multi-qa-MiniLM-L6 | 0.913 | 0.964 | 0.984 | **0.995** | **1.000** |

  The general `all-MiniLM-L6-v2` was worse (48/24 top-3 support 0.877, refute
  0.832), so `multi-qa-*` is the retriever. 64/32 is better than 48/24 on every
  axis *and* structurally guarantees containment up to 33 words
  (`W-S+1`), above the observed max of 32.
- **Conclusion:** adopt **64/32 + `multi-qa-MiniLM-L6-cos-v1` + top-8**;
  top-5 (0.964/0.984) is the latency fallback on the 1650. Top-3 alone
  (≈0.83–0.91 support) is **not** sufficient. Recorded before any training.

## T032/T033 — ModernBERT cross-encoder (learned localizer + decision)

- **Shape:** `[CLS] question [SEP] passage [SEP]` → 3-way SUPPORT/REFUTE/
  NOT_MENTIONED, token start/end, expected-tIoU. Trained on `evidence.csv`
  labels with cross-conversation NOT_MENTIONED negatives.
- **Protocol:** 5-fold grouped-by-conversation; held-out conversations scored
  through the *serving* code path (`predict_transcript`); COF/LOCO τ selection.
- **T032 (4 epochs):** OOF 0.546 native (acc 0.777, mIoU 0.391). Switching the
  decoder from argmax-SUPPORT+score-aware to **max-`p_support` + τ** (LOCO τ
  0.08) lifted it to **0.602** — the raw p is not temperature-calibrated, so the
  q-dependent cutoff was too strict.
- **T033 (inverse-frequency class weights, 6 epochs):** OOF native 0.609
  (acc 0.854, mIoU 0.446); **LOCO τ 0.16 → 0.610** (acc 0.859, mIoU 0.444),
  stable 0.15–0.16 across folds. By type: positive 0.800, hard_negative 0.873,
  off_topic 1.000; positive recall 0.800.
- **Compare legacy OOF (T030) 0.569** (acc 0.949, mIoU 0.316). ModernBERT trades
  ~9 points of accuracy for **+13 points of mIoU**, netting **+0.041**.
- **Fold spread (T033):** 0.652, 0.617, 0.663, 0.574, 0.522.
- **Conclusion:** the learned path beats the frozen legacy pipeline out of fold
  on the training set. Next: final model on all 39, 1650 latency/VRAM check,
  then flip `MEDAPP_ANSWERER` (validation only with approval).

## T034 — validation: served ModernBERT (0.606)

- **Config:** `MEDAPP_ANSWERER=modernbert`, `models/modernbert_final/final.pt`
  (trained on all 39, class weights, 6 epochs), `MEDAPP_MB_TAU=0.16`;
  `large-v3-turbo` int8 ASR; local GTX 1650 + cloudflared quick tunnel.
- **Results:** 19 validation conversations + verify, **all 200 OK, zero
  timeouts**; latency mean **20.3 s**, worst **32.4 s** (uncached ASR ~14–32 s,
  cached ~2.5–3 s); VRAM ~2.1 GB of 4 GB; **service score 0.606**.
- **Correspondence:** 0.606 vs the T033 OOF estimate 0.610 — the OOF predicts the
  service, as the legacy pipeline did.
- **Conclusion:** the ModernBERT path is the best validated config so far
  (legacy served 0.545, T027). The remaining loss is localization + decision
  paraphrase/slot errors; the hybrid (T035) is the next no-retrain gain.

## T035 — hybrid offline analysis (legacy decision + ModernBERT span)

- **Method:** offline join of the legacy large-NLI OOF (`results/oof_T030.json`)
  and the class-weighted ModernBERT OOF. Decide with the legacy NLI (merged
  clause premise + guard); take the span from ModernBERT. Two strategies, τ
  selected leave-one-conversation-out.
- **A (legacy decides):** LOCO **0.647** (acc 0.933, mIoU 0.456, τ 0.30);
  positive 0.944, hard_negative 0.930, off_topic 0.943.
- **B (`MB_yes OR legacy p≥0.80`):** LOCO **0.661** (acc 0.928, mIoU 0.483,
  τ 0.80); positive 0.979, hard_negative 0.852, off_topic 0.981.
- **Why it works:** legacy MNLI recovers 35/39 paraphrase misses and rejects the
  slot-contradiction hard negatives; ModernBERT supplies the tighter span
  (tIoU 0.557 vs legacy ~0.36).
- **Latency note:** only the legacy *decision* is needed (localization dropped,
  ~8.8 s saved), and it can be gated to questions ModernBERT answers no.
- **Status:** offline only; not yet implemented as an answerer or served. Next:
  `answerers/hybrid.py`, OOF calibration, 1650 latency check, validate.


## T036 — LLM ceiling probe (Qwen2.5-7B-Instruct, 39 conversations, in-sample)

- **Setup:** transcript-only, one call per conversation, strict JSON with a
  verbatim `evidence_quote` aligned to ASR words. L0 zero-shot, L1 few-shot
  (3 LOCO-safe balanced examples), L2 two-pass (decide, then cite). V100-32G,
  fp16, plain `transformers`; scripts `llm_probe.py`, `answerers/llm_*`.

| rung | score | acc | mIoU | pos recall | quote found | parse fails | latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.588 | 0.962 | 0.339 | 0.939 | 0.776 | 10 | 9-16 s |
| **L1** | **0.665** | **0.982** | **0.454** | **0.980** | **0.954** | **0** | 9-16 s |
| L2 | 0.611 | 0.985 | 0.362 | 0.980 | 0.855 | 0 | 12-24 s |

- **Few-shot is decisive:** L0 → L1 gains +0.077, driven by quoting
  (`quote_found` 0.776 → 0.954) and format (`parse_fail` 10 → 0), with accuracy
  0.962 → 0.982. L2's isolated quote pass hallucinates more than joint
  decide+cite (0.855 vs 0.954) despite the best accuracy (0.985).
- **Decision is near-perfect:** L1 positive 191/195, hard_negative 140/142,
  off_topic 52/53; only 4 missed positives and 3 false positives. L1 catches
  38 positives ModernBERT misses (MB catches 3 L1 misses).
- **Localization is the remaining loss:** mean tIoU when yes 0.464 (lower than
  ModernBERT's 0.557); 47/191 yes have tIoU < 0.1 — wrong occurrence of a
  repeated/paraphrased statement, not boundary width. Ceiling: in-sample.

## T037 — hybrid simulation: LLM decision + ModernBERT span

- **Method:** offline join of T036 L1 decisions with T033 ModernBERT OOF spans;
  span = ModernBERT span when it exists, else the L1 span.
- **Result:** **0.705** (acc 0.982, mIoU 0.521) vs L1 alone 0.665 and
  ModernBERT alone 0.609. The LLM supplies the near-perfect decision and
  ModernBERT the better span.
- **Caveat:** in-sample ceiling (both components are scored on the 39 labelled
  conversations). Not served. Localization (occurrence choice) is still the
  largest remaining gap (0.521 vs the 0.843 served-cap oracle).

## T038 — LLM L1 with Gemma 4 E4B (0.729, best measured)

- **Model:** `google/gemma-4-e4b-it` (~8B params, fp16 15.9 GB, V100-32G),
  plain `transformers`, text-only L1 prompt (whole transcript + 10 questions +
  3 LOCO-safe few-shot turns). Loaded via `AutoModelForImageTextToText` since
  the checkpoint is `Gemma4ForConditionalGeneration` (multimodal).
- **Result (39 conversations, in-sample):**

  | metric | Qwen2.5-7B (T036) | **Gemma 4 E4B** |
  | --- | --- | --- |
  | score | 0.665 | **0.729** |
  | accuracy | 0.982 | **0.990** |
  | mIoU | 0.454 | **0.555** |
  | tIoU when yes | 0.464 | **0.567** |
  | quote found | 0.954 | **1.000** |
  | positive | 191/195 | 191/195 |
  | hard_negative | 140/142 | **142/142** |
  | off_topic | 52/53 | **53/53** |
  | latency | 11.9 s | 21.2 s |

- **Localization:** worst-case (tIoU<0.1) dropped **47 → 22**; `>=0.8` rose
  43 → 55; 36 questions improved from tIoU<0.5 to >=0.5. Quote-found 1.000
  (no unfindable quotes). Remaining 22 include degenerate gold spans
  (`Good morning,`), alternate valid occurrences, and a few truly unsupported.
- **Caveats:** in-sample ceiling (few-shot from other conversations only,
  LOCO-safe), not validation. E4B fp16 needs 15.9 GB; int4 (~4 GB) is borderline
  for the 4 GB 1650, E2B would fit. E4B also accepts **native audio** — an
  untried angle.
- **Conclusion:** a small modern instruction model beats both the larger older
  Qwen2.5-7B and the 0.705 hybrid, and is close to a servable size. Next:
  validate (needs serving) or use as teacher; test E2B/int4 for the 1650.

## T039 — validation: served Gemma 4 E4B L1 (0.744)

- **Config:** `MEDAPP_ANSWERER=llm`, `MEDAPP_LLM_MODEL=google/gemma-4-e4b-it`
  (fp16, `AutoModelForImageTextToText`), L1 whole-transcript prompt with 3
  LOCO-safe few-shot turns; `large-v3-turbo` int8 ASR; IDUN A100/H100 node +
  cloudflared quick tunnel.
- **Results:** 19 validation conversations, all 200 OK; latency **14–19 s**;
  **service score 0.744**.
- **Correspondence:** in-sample probe T038 = 0.729; validation 0.744 (the probe
  is again a faithful, slightly conservative predictor of the service).
- **Bug fixed en route:** `example.py` built a fresh `LLMAnswerer` per request,
  reloading the 16 GB model each time (36–48 s, one failure). `build_answerer`
  is now a process-wide singleton → 14–19 s.
- **Compare:** ModernBERT served 0.606 (T034), legacy served 0.545 (T027).
- **Conclusion:** the best validated config so far by +0.14 over T034. Remaining
  risk is serving stability for the one-shot evaluation (ephemeral tunnel,
  4 h job walltime). In-sample probe predicts the service.

## B1 — native-audio probe (in progress)

- **Goal:** ask Gemma 4 E4B to transcribe the audio directly and see whether it
  fixes the Whisper medical-term errors (PAMEL→Pamol, ibumedin→Ibumetin,
  panadil→Panodil, Activel→Activelle, Isomeprazole→Esomeprazole,
  "long-term sugar value"→HbA1c) and whether the latency is affordable.
- **Blocker:** the multimodal generation path needs **torch>=2.6**
  (`masking_utils` `or_mask_function`); IDUN `nordic` has torch 2.5.1. Needs a
  separate env (`torch>=2.6`) or an older transformers, so as not to disturb the
  serving env / the 0.744 endpoint.
- **Script:** `llm_audio_probe.py` (decodes MP3 via
  `faster_whisper.audio.decode_audio`, feeds `<|audio|>` + `input_features`).

## T040 — LLM L1 with Gemma 4 26B-A4B (0.755 in-sample)

- **Model:** `google/gemma-4-26b-a4b-it` (26B MoE, 3.8B active; fp16 51.6 GB;
  H100/H200 80 GB node, `idun-06-04`), same L1 prompt/harness as T038.
- **Result (39 conversations, in-sample):** score **0.755**, acc **0.997**,
  mIoU **0.594**, positive recall 0.995, quote found 1.000, 0 parse fails,
  latency mean 16.6 s / worst 19.2 s.
- By type: positive 194/195, hard_negative 142/142, off_topic 53/53 (only **1**
  decision miss vs E4B's 4).
- **Scaling gains vs E4B (T038):** mIoU 0.555 → 0.594, `>=0.8` spans 55 → 63,
  23 questions moved from tIoU<0.5 to >=0.5. **But the 22 worst cases
  (tIoU<0.1) are identical** — a structural floor (ambiguous occurrence +
  degenerate gold), not capacity.
- **Interpretation:** model scale buys decisions and middle-quality spans, with
  diminishing returns; the remaining loss needs occurrence supervision, not
  more parameters.
- **Serving:** 26B fp16 holds an 80 GB node; E4B (validated 0.744, 14–19 s)
  stays the default unless 26B is chosen for the evaluation.

## Loop baseline smoke — baseline-smoke-6f41260d

- **Observed incumbent:** the active IDUN process runs Gemma 26B L1/base,
  turbo int8 ASR, not E4B. The historical 0.744 validation belongs to E4B;
  it must not be attributed to this process.
- **Isolation:** `idun/run.py` snapshots source/configuration into a separate
  run directory, copies only supplied-training transcripts, and refuses a
  second active experimental job. No live files or captures are synchronized.
- **Smoke:** job `25405036`, A100 80 GB, revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Thirty questions from three
  supplied conversations: accuracy 1.000, mIoU 0.7104, score 0.8262,
  no parsing/alignment failures. All ASR was cached; request latency
  17.90 s mean / 18.31 s maximum. This is a plumbing gate, **not** a
  full-corpus improvement or an uncached latency result.
- **Cold load:** model loading alone took over five minutes on this allocation.
  Readiness must wait for actual warm-up, and recovery must return bounded
  guesses while the owned worker reloads.
- **Tokenizer audit:** the pinned E4B and 26B tokenizers produce identical
  token IDs with and without added special tokens. The general duplicate-BOS
  risk is not an observed defect for these checkpoints.
- **Measurement corrections:** retain question and passage provenance and split
  before negative generation; save proposed spans even for no decisions; decode
  ordered, non-padding spans; reject ungrounded evidence instead of citing a
  whole passage; count unsent conversations after abort; restrict candidate
  oracles and alignment to legal cited regions.
- **Next measurement:** complete, uncached 26B/turbo baseline followed by
  matched-transcript v2/scoped prompt comparisons. Boundary calibration excludes
  demonstration-source conversations from its cross-validation pool.

## Loop full baseline and boundary calibration — full-grounding-f8bf809f

- **Baseline stage complete:** 39 conversations / 390 questions through
  `example.predict`, 26B/base revision as above, turbo int8, uncached ASR for
  every conversation, output cap 1024. Score **0.7886**, accuracy **0.9974**,
  mIoU **0.6494**, positive recall 194/195; all hard negatives and off-topic
  questions correct. No parse/alignment fallbacks. In-process latency mean
  **18.81 s**, p95 **23.02 s**, maximum **23.63 s**.
- This measures the actual incumbent configuration more faithfully than the
  old large-v3 probe. It is **not** a new official validation score, an HTTP
  latency gate, or a claimed improvement caused by changing the live service.
- **Offset hypothesis:** fit a predeclared 64-pair grid of small constant
  boundary corrections, keeping decisions fixed. Exclude every conversation
  used as a demonstration anywhere in the input records (`sample_10`,
  `sample_17`, `sample_18`, `sample_19`) before grouped calibration.
- **Result:** on the remaining 35 conversations / 350 questions, five-fold
  score **0.7859 -> 0.7978** (mIoU **0.6452 -> 0.6650**), unchanged accuracy.
  Both seeds 13 and 37 select **start +0.2 s, end +0.0 s** in every fold.
  Paired conversation-bootstrap 95% intervals for score gain:
  **[+0.00715, +0.01701]** and **[+0.00721, +0.01669]**.
- **Decision:** retain the offset candidate for an uncached HTTP gate. The
  artifact is bound to ASR configuration `e75a7f6e`, the exact LLM revision,
  base prompt, and input-record hash. A correction that collapses a short
  citation preserves its original span. The live endpoint remains unchanged.
- **Prompt comparison completed:** minimal-quote v2 scored **0.7740**
  (delta -0.01457, bootstrap interval [-0.03900, +0.00743]); segment-scoped
  scored **0.7681** (delta -0.02051, interval [-0.04690, +0.00319]), with
  three alignment failures and 26.48 s mean cached latency. Neither
  demonstrates improvement; both are rejected for deployment. Keep base.
- **Next gate:** `offset-http-be360d39`, job `25405044`, runs the offset
  candidate over the complete corpus through an owned loopback HTTP endpoint
  with ASR caching disabled. Its source and calibration artifact are frozen,
  and inputs are copied from the completed comparison. No live change yet.

## Offset release qualification and publication

- **Normal-order HTTP:** `offset-http-be360d39`, score **0.8007415**,
  accuracy **0.9974359**, mIoU **0.6696119**, 390 questions, no failures or
  timeouts, p95 **21.93 s**, maximum **23.81 s**, ASR caching disabled.
  Every prediction exactly matched the baseline decision and intended
  boundary transform. No unexpected span changes or runtime fallback logs.
- **Question-order confirmation:** `offset-shuffle-e67b7a89`, score
  **0.8045887**, same accuracy, no failures/timeouts, maximum **22.03 s**.
  This is a robustness check, not a reason to optimize a shuffle seed.
- **Release:** `offset-release-dns-2732ab44`, job `25405052`, repeated
  the full gate at **0.8007415**, maximum **23.88 s**. External root and
  ten-question prediction checks returned 200; predictions matched the
  local trace, round trip **20.48 s**. The qualified endpoint is recorded in
  its `endpoint.json`; no competition attempt or settings were changed.
- **Publication failure and fix:** the first release passed inference gates
  but fresh-host DNS returned NXDOMAIN from IDUN. A CPU nonce-only tunnel
  probe reproduced that while the external controller reached the same
  tunnel successfully. Publication now waits for an external root and actual
  prediction check, allows DNS propagation, and retains TLS verification.
  The first failed release was stopped; the original service was untouched.

## Rejected CPU refinements

- **Anchored delta-IoU ridge**, `span-ridge-53acfa67`: candidates within the
  calibrated quote +/- 4 words plus the unchanged citation; 24 runtime-only
  features, actual candidate-minus-baseline tIoU targets, nested grouped
  policy selection, demonstration conversations excluded. Candidate oracle
  mIoU **0.7978**, but scores **0.7958 / 0.7964** versus retained **0.7978**
  for seeds 13/37. Only 5/7 changes, with negative mean deltas and intervals
  including harm. **Rejected; no runtime integration.**
- **Fine offsets**, `fine-offset13-09e841c5`: start grid 0.10–0.30 and end
  -0.10–0.10 in 0.05-second increments. Full-data fit preferred +0.25/0,
  but grouped score fell **0.7978 -> 0.7977**; delta -0.000134,
  interval [-0.001522, +0.001312]. **Rejected at the first gate**, without
  changing the retained +0.2/0 artifact.
- **Residual evidence:** calibrated mIoU 0.6696; gold-assisted within-quote
  oracle 0.7511, quote +/- 24 words 0.8763, raw-word oracle 0.9273.
  Best-of-base/v2 span oracle 0.7137, all three prompts 0.7307.
  These are ceilings, not achieved improvements. Selection still matters more
  than medical word recognition.

## Prepared full-context demonstration ablation

- **Hypothesis:** the positive few-shot currently shows only a gold-centered
  excerpt, hiding alternative supporting occurrences elsewhere in the same
  conversation. `full_context` reveals that complete positive-example
  conversation while keeping its question, reference answer, demonstration
  count/source selection, negative examples, and target output schema unchanged.
- **Run:** `full-context-bbcac1a3`, job `25405056`, is dependency-blocked on
  original serving job `25404584`. It holds **no GPU allocation while pending**.
  The qualified offset endpoint stays running; the old service is not stopped
  early. Comparisons require identical audio/transcript hashes and report both
  raw and fixed +0.2-second policies on demonstration-disjoint questions.

- **Outcome:** completed after the old allocation's natural expiry. Raw score
  0.7895 vs 0.7886, but fixed-offset score **0.8005 vs 0.8007**.
  Demonstration-disjoint delta **-0.000284**, interval
  **[-0.008549, +0.008071]**; decisions unchanged, identical transcript hashes,
  no parse/alignment failures. **No reliable gain; rejected for deployment.**
- **Next controlled change:** `two-positive-dbbb3183`, job `25405084`, adds
  exactly one positive demonstration, keeping the base excerpt representation
  and one hard-negative/one off-topic example (counts 2/1/1). Reference
  comparisons exclude the union of all demonstration-source conversations.
  The qualified release remains unchanged.

- **Additional-positive outcome:** rejected. Raw score **0.7773** and
  fixed-offset score **0.7876**, with one additional missed positive overall.
  On the common 34-conversation non-demo cohort, fixed-offset delta
  **-0.011869**, interval **[-0.029378, -0.000398]**. Additional examples did
  not improve localization; the base demonstration policy stays unchanged.

## Turbo precision ablation — turbo-fp16-65db8c47

- **Hypothesis:** int8 was selected to fit the old 4 GB host. Test whether
  float16 changes downstream word alignment/citation quality on the permitted
  80 GB target, keeping the LLM revision and base prompt unchanged.
- **Run:** job `25405086`, uncached ASR over all 39 clips. The harness records
  CTranslate2's actual compute type/device and requires `float16`, so a fallback
  cannot masquerade as this experiment.
- Audio hashes must match the baseline. Transcript differences are explicitly
  labeled as the tested ASR axis rather than bypassing method-comparison
  checks silently. Compare raw and fixed-offset scores on the common
  demonstration-disjoint cohort; do not change the live int8 release yet.

- **Outcome:** actual CUDA float16 verified, 390 questions, same 0.9974
  accuracy. Raw score **0.7941**, transferred-offset score **0.8062**,
  maximum uncached latency **22.66 s**. The non-demo offset delta is
  **+0.006140**, but its interval **[-0.005811, +0.019742]** includes harm.
  **Promising point estimate, not qualified for promotion.**
- **Crossover diagnosis**, `precision-cross-6c5bf9c2`: word text was identical
  in 13/39 conversations; word timestamps differed in all 39. Keeping int8
  quotes and swapping matched fp16 timings gave only **+0.000607**
  (inconclusive). Fp16 quotes mapped back to int8 timings gave **+0.007395**
  (also inconclusive). Separate fp16 grouped calibration still chose +0.2/0
  in every fold for both seeds. The point-estimate gain mainly accompanies
  changed citation selection, not a demonstrated timing improvement.

## Numeric-timestamp serialization ablation — no-timestamps-7cbd936f

- The model outputs quotes, and code supplies their timestamps. Test whether
  redundant numeric time headers affect evidence selection: retain segment IDs,
  words, order, demonstrations, and output schema, but omit the time numbers
  from target and demonstration text.
- Job `25405091` uses the frozen int8 transcripts and the same 26B revision.
  Audio/transcript hashes must match. Compare raw and fixed-offset outcomes
  on the common non-demo cohort; keep the qualified endpoint unchanged.

- **Outcome:** rejected. Raw score **0.7818**, fixed-offset score **0.7939**,
  one additional alignment failure/missed positive. Non-demo fixed-offset
  delta **-0.007461**, interval **[-0.022315, +0.007240]**. Retain timestamped
  base serialization.

## Anchored localization-only refinement — local-refine-5fa55487

- **Different from earlier L2:** freeze all baseline decisions and refine only
  positive citations within the current quote +/-24 words. Two keep/refine
  demonstrations come only from the already excluded support pool. Every
  predicted positive is attempted regardless of whether gold lies in its
  region; missing, ambiguous, or unaligned proposals retain the baseline.
- **Run:** job `25405097`, same frozen 26B revision, at most 20 seconds for
  the added call. The probe records actual added latency and a sum with the
  earlier uncached baseline timings; that sum is only an estimate, not an
  end-to-end HTTP result.
- Require non-demo paired improvement before building a serving path. The
  published base+offset model remains unchanged.

- **Outcome:** rejected. The model kept **191/194** positive citations and
  refined three. Score **0.8003 vs 0.8007**, unchanged decisions; non-demo
  delta **-0.000703**, interval **[-0.002158, 0]**. Added latency peaked at
  14.46 s; the maximum sum with earlier baseline latency was 38.09 s, still
  only an estimate. No generation failures, but no quality gain to justify
  extra inference or serving integration.

## Preregistered evidence fusion check

Before spending on more model calls, use saved base/v2 outputs to test 18
interpretable policies: unchanged baseline; endpoint blends and shorter/longer
selection gated by overlap; earlier/later preference only for disjoint spans.
Base decisions remain fixed. Choose policies on training conversations and
evaluate held-out conversations, excluding all demonstration sources, for
seeds 13 and 37. Require positive paired evidence in both seeds; estimated
two-call latency is not an HTTP acceptance gate.

- **Outcome:** `evidence-fusion-f7f3eeff` rejected. Non-demo scores
  **0.7976 / 0.7974** versus retained 0.7978; mean deltas negative for both
  seeds, no positive confidence bound. Only 7/4 questions changed. No
  evidence justifies the second model call.

## Bounded capacity smoke preparation

After several controlled prompt/refinement failures, test a different capacity
axis rather than repeating them. Candidate: `google/gemma-4-31B-it`, pinned
revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. The official card describes
a dense model, unlike the incumbent MoE, so latency is a material risk.
Start with only three conversations on one 80 GB GPU; a full run needs the
memory, latency, and quality smoke gates first.

The weight manifest is 62,546,338,248 bytes. Own-user quota was checked before
preparation: approximately 203,736,996 KiB used against a 1,000,000,000 KiB hard
limit. Model caching is development-only, revision- and size-bounded, and does
not upgrade the shared serving environment or enable inference network calls.
Primary model card: `https://huggingface.co/google/gemma-4-31B-it`.

- **Cache verified:** `cache-dense31-ae5f11b8`, all nine selected files,
  62,578,656,403 bytes including tokenizer/configuration artifacts.
- **Smoke:** `dense31-smoke-b824cc20`, job `25405103`, native BF16 as specified
  by the pinned checkpoint. A100 80 GB, 62.57 GB torch-resident model memory.
  Thirty questions: accuracy 1.000, no parse/alignment failures, raw score
  0.8022, cached latency 19.80 s mean / 20.86 s maximum. The incumbent's
  three-conversation raw score was 0.8262; this is a capacity/format gate, not
  a demonstrated quality gain.
- Memory and latency permit one complete matched-input comparison. No
  deployment claim is justified by the three-conversation smoke.

- **Full outcome:** `dense31-full-296207d2`, job `25405104`, rejected.
  Raw score **0.7864**, fixed-offset score **0.7978** versus incumbent 0.8007;
  192/195 positives correct, one alignment failure. Non-demo fixed-offset
  delta **-0.001458**, interval **[-0.032379, +0.025163]**. Memory and speed
  were viable, but capacity alone did not improve annotation occurrence choice.

## Conditional final-statement occurrence rule

The next isolated prompt change tests a concrete residual error mechanism:
when several passages consistently establish the same queried fact, prefer its
final specific statement or confirmation. Preserve subject, temporal status,
dose, and qualifiers; later contradictory statements or generic acknowledgements
are not substitutes. Model, demonstrations, ASR and output schema stay fixed.
No question-specific or timestamp-specific rules are introduced.

- **Outcome:** `final-statement-9a77a11e` rejected. Fixed-offset score
  **0.7938**, one added alignment failure/missed positive; non-demo delta
  **-0.007585**, interval **[-0.019900, +0.001271]**. A generic later-statement
  preference is not a reliable substitute for reference occurrence selection.

## Preregistered blind local extraction

The earlier anchored refiner kept 191/194 citations. Test whether showing the
existing answer over-anchors generation: retain the same +/-24-word regions,
base decisions, excluded demonstration pool, and 20-second added-call cap,
but hide the current quote and request an independent exact local citation.
Reference examples emit quotes rather than keep/edit labels. Invalid or
ambiguous proposals still retain the baseline. No live integration without
paired non-demo improvement and a real end-to-end latency gate.

- **Outcome:** `blind-refine-eeda139d` rejected. It changed 107 citations
  rather than three, but score fell to **0.7825**; non-demo delta
  **-0.018305**, interval **[-0.054244, +0.012678]**. Decisions remained fixed
  and no generation failed. Removing anchoring caused more edits, not better
  annotation alignment.

## Supervised reference-localizer feasibility

Prompt-only refiners and unsupervised capacity changes have not learned the
reference citation convention. Prepare a bounded supervised E4B quote-localizer
pilot with frozen base weights and rank-4 attention q/v adapters. Preserve the
26B decisions; evaluate every baseline-predicted positive, not a gold-selected
applicability subset. Use an excluded demonstration pool and conversation-held-out
fold 0 of five (seed 13), two fixed epochs, before considering full OOF work.

PEFT and Accelerate were absent. Training prerequisites are installed only in a
run-local system-site-packages virtual environment, constrained to the exact
existing torch/transformers/ASR runtime versions. The live `nordic` environment
is never upgraded. Require dependency, import, single-step gradient/memory,
and unadapted-versus-adapted comparison gates before spending on full training.
Primary adapter documentation: `https://huggingface.co/docs/peft/v0.21.0/en/package_reference/lora`.

- **Environment verified:** `lora-env-915fada0`, isolated PEFT 0.21.0 and
  Accelerate 1.15.0; torch 2.5.1+cu121, transformers 5.17.0 and both ASR
  packages remain unchanged. Text attention q/v projections were identified
  from cached tensor headers without loading another inference model.
- **Data preflight:** causal masking uses the same rendered chat-template
  prefix as inference and supervises completion tokens only; overlength
  examples fail rather than truncating evidence. Demonstration-source and
  held-out conversations are excluded before forming training targets.
- **Next gate:** one optimizer step on the longest training sequence, BF16
  E4B, rank 4 / alpha 8 q/v adapters, dropout 0.05, learning rate 1e-4.
  Require finite loss, nonzero finite adapter gradients, frozen base weights,
  and measured memory headroom. The pilot remains fixed at two epochs,
  accumulation 4, and conversation-held-out fold 0 / seed 13.

- **Gradient smoke passed:** `lora-gradient-smoke-8271b5b0`, A100 80 GB,
  143 training-only examples, longest sequence 2,031 tokens, 1,134,592
  trainable adapter parameters. Finite loss 0.48934, finite nonzero gradients,
  peak allocation **25.05 GB**; one update took 3.08 s. Base parameters stayed
  frozen. Add an adapted-generation check on a training-only example before
  the held-out pilot; this does not select a checkpoint using held-out labels.

- **Adapted generation passed:** `lora-generation-smoke-efcaa693`, same
  finite loss and 25.05 GB peak. After the optimizer update, the adapter
  produced valid JSON with a verbatim quote found in its training-only
  probe transcript. Proceed to the fixed two-epoch, fold-0 pilot; do not
  interpret these feasibility checks as a generalization result.

- **Fixed held-out pilot:** `lora-fold0-pilot-4af0ceec`, 7 conversations /
  70 questions. Baseline **0.8362**, unadapted E4B localizer **0.7650**,
  adapted **0.8176**. Relative to unadapted, delta **+0.05258** with interval
  **[+0.02253, +0.08519]**; relative to the incumbent, delta **-0.01865**
  with a wide interval. All 35 predicted positives localized; decisions
  unchanged and no alignment/budget fallback. Estimated combined maximum
  28.11 s is not a serving measurement.
- Learning is measurable, but this fold does not beat the incumbent and
  cannot justify deployment. Complete the remaining four fixed folds
  sequentially on one GPU, reusing (not reselecting) the frozen fold-0
  result. Verify source hashes, identical hyperparameters, split isolation,
  complete unique OOF coverage, and unchanged base decisions.

- **Complete OOF outcome:** `lora-oof-e9bb2586`, 350 non-demo questions
  covered exactly once. Baseline **0.7978**, unadapted **0.7709**, adapted
  **0.7976**. Adaptation improves its own model by **+0.02673**
  (interval **[+0.00845, +0.04609]**) but does not beat the incumbent
  (delta **-0.000226**, interval **[-0.02542, +0.02483]**).
  No standalone deployment or full-data training is justified.
- **Fixed agreement diagnosis:** evaluate exactly two predeclared rules using
  these OOF proposals: use the adapter only when span tIoU with the incumbent
  is at least 0.5, or average endpoints under the same gate. Otherwise keep
  the incumbent. No policy/threshold fitting on OOF features is performed,
  avoiding cross-fold meta-training leakage. Report both rules honestly.

- **Agreement outcome:** neither rule qualifies. Adapter-on-agreement delta
  **+0.001286**, interval **[-0.006953, +0.009396]**; midpoint delta
  **-0.000026**, interval **[-0.004391, +0.003869]**. Keep the incumbent.
- **Next bounded diagnostic:** use each frozen outer-fold adapter's mean
  supervised-completion NLL to choose between its own grounded OOF proposal
  and the incumbent quote. Only differing proposals are scored; no reference
  quote is supplied as an inference candidate, no threshold is fitted, and
  base decisions remain fixed. Record model-scoring overhead separately.

- **Likelihood outcome:** `adapter-nll-e2e0d52d` remains unqualified. It
  preferred the adapter in 54/57 differing proposals; score 0.8019 versus
  0.7978, delta **+0.004080**, interval **[-0.019149, +0.027775]**.
  Scoring added at most 0.99 s per compared question, but uncertainty does not
  justify adding the serving model.

## Training-target boundary audit

Supervised targets currently include every word that overlaps a reference
interval, potentially adding a word that overlaps by only a few milliseconds.
Audit the exact contiguous word range maximizing official tIoU after the
retained +0.2-second correction. Report both the unconstrained oracle and
uniquely decodable quotes, since the runtime preserves baseline evidence for
ambiguous text. Original reference timestamps stay unchanged; this is a
training-label derivation diagnostic, not achieved inference quality.

- The first audit exposed an infeasible target, not a reason to drop a row:
  `sample_64_yes_q02` has reference [0, 0.16]. The only positive-overlap
  corrected quote is the repeated word “Good”; a longer unique quote starts
  at 0.2 and cannot overlap the reference. Record zero with an explicit
  infeasibility flag and retain the question in every denominator.

- **Complete audit:** `target-audit-complete-379b6c2b`, all 195 positives
  retained. Unique optimal targets differ on 20 examples (17 non-demo);
  target tIoU rises only **0.91587 -> 0.92526** overall and
  **0.91426 -> 0.92350** non-demo. One infeasible case remains zero.
  The predeclared 0.02-gap gate failed, so no target-only retraining is
  justified. Original annotations are unchanged.

## Native inference precision

The cached 26B configuration declares BF16; the existing inference recipe uses
FP16 inherited from earlier GPU compatibility. Test BF16 on the same 80 GB
target while fixing the model revision, int8 transcripts, base prompt and
decoding. Record and assert actual parameter dtype; compare common non-demo
raw/fixed-offset results. Do not change the live FP16 release on a point
estimate alone.

- **Outcome:** `incumbent-bf16-ef8602bb` rejected. Actual BF16 verified with
  matched transcripts and unchanged decisions, but fixed-offset score
  **0.7924** versus 0.8007; non-demo delta **-0.009161**, interval
  **[-0.019913, -0.001105]**. Preserve FP16 for this checkpoint.

## Local word-pointer representation

Use the blinded local-extraction setup, fixed decisions, same +/-24-word
regions and excluded demonstration sources, but request inclusive local word
IDs rather than generated quote prose. Reference examples use the same
overlap-derived word ranges. Reject invalid or out-of-region indices and retain
baseline evidence; keep the 20-second added-call cap. This tests representation,
not a new global occurrence preference or label-dependent applicability gate.

- **Outcome:** `local-word-pointer-b6204400` rejected decisively. It returned
  valid ranges and kept decisions fixed, but full score dropped to **0.7347**.
  Non-demo delta **-0.072250**, interval **[-0.107292, -0.041332]**.
  All 194 citations changed. Index validity is not evidence quality.

## Deterministic beam-width smoke

Test beam width 2 rather than greedy decoding with the exact incumbent
checkpoint, FP16, frozen int8 transcripts, base prompt and response schema.
First allow only three conversations on one 80 GB GPU. Require memory,
format and latency headroom before a full comparison; no stochastic seed
selection and no live change. Beam count is recorded in run metadata and
defaults to 1 for every existing path.

- **Smoke gate:** `beam2-smoke-c7d1b915`, actual width 2, all 30 decisions
  correct and no parse/alignment failures. Raw score **0.8264**, identical
  to the exact matched-transcript greedy subset; maximum cached latency
  **27.67 s**. The earlier 0.8262 smoke used older cached transcripts and
  is not this comparison's reference. Headroom permits one full paired
  evaluation, but the smoke demonstrates no quality gain.

- The first full-run submission had a mistyped reference-manifest path and
  was cancelled before inference. Reference files are now loaded before model
  warm-up so invalid experiment inputs fail without costly weight loading.
  The cancelled run is not counted as a scientific result.

- **Full outcome:** `beam2-full-fixed-d4159b42`, all 390 questions. Raw
  score 0.7886 and fixed-offset **0.8007415** exactly tie greedy decoding;
  the common non-demo paired delta and its interval are both zero.
  Cached latency averages 24.58 s, maximum 29.39 s. Reject the slower
  decoder; no serving change.

## Metric-reward grounding: research comparison, September 19, 2026

The target is composite score near 1.0, not mIoU 0.8007. The qualified local
result remains accuracy 0.9974, mIoU 0.6696, composite 0.8007. At that accuracy,
composite 0.95 requires mIoU about 0.9184. Neither an oracle nor a training-set
reward is an achieved held-out score.

| Primary source | Difference from our current method | Bounded adaptation |
| --- | --- | --- |
| [Time-R1](https://arxiv.org/abs/2503.13377), June 29, 2025 revision | Verifiable temporal rewards optimize localization rather than token likelihood. | Warm-start from the already trained, conversation-disjoint SFT adapter; optimize decoded intervals. |
| [SelfCite](https://arxiv.org/abs/2502.09604), June 15, 2025 revision | Citation quality uses context removal/retention, not the quote's generation probability. | The rejected quote-NLL reranker did not implement this objective; do not label that result a SelfCite test. |
| [EvoGround](https://arxiv.org/abs/2605.13803), May 13, 2026 | Coupled proposer/solver agents create grounding supervision from unlabeled video. | Possible later training-data axis, not permission to generate or alter held-out labels. |
| [TimeLens2](https://arxiv.org/abs/2607.17423), July 19, 2026 | GRPO combines tIoU with exact temporal Wasserstein distance, distinguishing some zero-overlap near misses. | Implement its interval geometry for this task's single span, while retaining the official scorer unchanged. |
| [Binary Discriminative Temporal Grounding](https://arxiv.org/abs/2608.08315), August 7, 2026 | Training-free, coarse-to-fine window verification avoids direct timestamp generation. | A later candidate-selection axis; our previous topic-overlap NLI is not a reproduction of its frozen video-language verifier. |
| [Qwen3 forced aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) | Dedicated audio/text alignment instead of relying only on ASR word times. | Separate timing experiment; an aligner cannot repair selection of the wrong supporting occurrence by itself. |

These are verified primary sources available before the review date, not a
claim that a video benchmark's SOTA score transfers to medical audio.
TimeLens2's correct identifier is **2607.17423**; the earlier search result
pointing at 2607.15424 was unrelated and was not used.

The implemented pure reward is official single-span tIoU plus
`exp(-W1 / (gold_duration + 1e-8))`, with -1 for invalid/ambiguous grounding.
The one-dimensional W1 is exact for uniform distributions on single intervals.
Decoding uses the same unique-quote match, +0.2-second start correction and
short-span collapse behavior as the SFT localizer. In particular, invalid
rollouts never collect the reward of an inference fallback.

Use pinned TRL 1.13.0 and datasets 5.0.1 in a **new run-local overlay**;
protect the installed torch, transformers and ASR stack. First verify API
compatibility, real nonzero adapter updates, sampled reward variation and
memory on one GPU. Then evaluate a fixed held-out fold with all questions
retained and unchanged 26B decisions. No SOTA reproduction, score gain,
full-OOF result or deployment is claimed by this implementation alone.

- **Dependency gate:** `grpo-env-b96d8c5a` failed before model loading:
  TRL 1.13.0 imports `FSDPModule`, absent from shared torch 2.5.1.
  The next fresh overlay explicitly pins torch 2.7.1/cu126 and matching
  vision/audio wheels. The owned node's driver is 575.57.08; no FSDP import
  monkey patch or shared-package upgrade is used. Setup verifies the base
  environment still has its original versions and records both runtimes.
  Pre/post-training SFT replay must therefore use this same training runtime.

- **Implemented fixed pilot:** one GRPO epoch, four samples per training
  question, lr 1e-5, temperature 1, KL coefficient 0.02 against a frozen
  copy of the SFT adapter, Dr.GRPO loss without per-group reward scaling,
  128 completion tokens, no vLLM or extra GPU. First run eight updates
  on the longest training prompts as a feasibility smoke; never evaluate or
  select a checkpoint using held-out rewards.
- Preflight checks every baseline question/reference, transcript hash, exact
  SFT training/held-out/demo split, adapter model/revision, and prompt tokens.
  Rewards receive only training IDs, not held-out rows. Raw sampled rewards
  and grounding failures are journaled; finite gradients, actual parameter
  movement, nonconstant rewards and an unchanged SFT KL reference are gates.
  Held-out replay includes all 70 fold-0 questions with frozen decisions.
- A stopping-token audit found tokenizer EOS 1 but native generation stops
  `[1,106,50]`. TRL masks completions using one tokenizer EOS. Derive that
  EOS from the actual SFT assistant template instead of accidentally masking
  completed turns as truncated; verify prompt-token parity afterward.
- **Compatible overlay passed:** `grpo-runtime-9c9e0e23` imported the pinned
  trainer/config and verified torch 2.7.1/cu126 inside the overlay, original
  torch 2.5.1/cu121 in serving, and unchanged transformers/ASR versions.
  Local real-artifact preflight covers 143 training positives, 70 held-out
  questions and all 39 content-hashed transcripts. GPU behavior remains a
  separate gate.
- **GRPO feasibility passed:** `grpo-gradient-smoke-be8cf4c9`, eight real
  updates / 32 rollouts. Seven groups had nonconstant rewards, 29 rollouts
  grounded uniquely, 26 overlapped their training reference. Adapter movement
  L2 0.04160, finite nonzero gradients, 17.48 GB peak, training 49.73 s;
  post-update greedy generation grounded successfully. The SFT KL reference
  remained bit-identical. These are feasibility measurements, not held-out
  score improvement.
- Restore native inference generation settings after GRPO: the trainer
  aligns its model config to the single training EOS, whereas pre/post
  held-out replay must retain identical original stopping behavior.
  Proceed to the one-epoch fixed fold-0 pilot from the **original** SFT
  checkpoint, not the eight-update smoke adapter.
- **Independent timing preparation:** `cache-qwen-aligner-b4ced968` cached
  all six required native-HF files, 1,847,225,867 bytes, at official revision
  `c07281df297b9905d24a508279258cccf987a064`. Native alignment prepare/decode
  APIs are present in transformers 5.17.0. This CPU-only preparation made no
  predictions and does not establish timing accuracy or request latency.
- **Held-out pilot:** `grpo-fold0-pilot-caae4435`, job `25405990`,
  started from the original SFT adapter with the fixed hyperparameters above.
  The A100/BF16 setup is held constant for comparability, not because the
  measured 17.48 GB smoke peak requires 80 GB.
- **Pilot result:** all 70 questions scored, unchanged decisions, all 35 yes
  outputs localized. GRPO composite **0.8234**, mIoU **0.7057**; SFT
  composite 0.8176, incumbent 0.8362 on this same fold. The paired GRPO-SFT
  delta is **+0.005823**, interval **[-0.015673, +0.025508]**; against the
  incumbent **-0.012827**, interval **[-0.086988, +0.059290]**.
  Five SFT spans changed; replay of the original SFT predictions was
  pointwise identical in the new runtime. Training used 572 rollouts, 101
  variable-reward groups, 17.48 GB peak and 597 s.
- This pilot is inconclusive, not a new qualified score. Complete the other
  four fixed folds serially using their existing SFT adapters and the exact
  same source/runtime/hyperparameters, reusing the fold-0 result rather than
  selecting a new seed or checkpoint. Require all 350 non-demo questions
  exactly once and preserve every baseline decision. Do not deploy or tune
  on this pilot; the healthy published service remains unchanged.
- **Complete OOF submitted:** `grpo-oof-bc86d18e`, job `25406077`, reuses
  fold 0 and runs folds 1-4 sequentially on the single experimental GPU.
  All frozen pilot source hashes and fixed hyperparameters matched before
  submission. Its monitor is `grpo-oof-watch`; no extra GPU experiment
  should be submitted while it is active.
- **Complete GRPO OOF outcome:** all 350 non-demo questions appeared once
  with unchanged decisions. GRPO composite **0.8040**, mIoU **0.6753**,
  versus incumbent **0.7978**, mIoU **0.6650** on this same cohort.
  The paired composite delta is **+0.006181**, interval
  **[-0.022475, +0.034658]**: not a qualified replacement.
  Same-runtime SFT scored 0.7986; the GRPO-SFT delta is **+0.005450**,
  interval **[-0.004435, +0.016439]**. One SFT replay span in fold 2
  changed under the newer runtime; that change is not attributed to RL.
  GRPO differs from the original SFT on 20 spans, with one ambiguity
  fallback. Maximum estimated combined time 34.89 s is not an HTTP gate.
  Retain the published incumbent; these 350-question scores must not be
  compared directly to its 390-question 0.8007 result.
- Inspection of the frozen GRPO rollout logs found 712 groups, 238 with
  identical proposed spans and 49 with zero tIoU throughout. Only five
  groups had diverse, grounded, disjoint spans but a distance-reward range
  at most 1e-8; 31 individual distance rewards underflow when cast to float32.
  This exposes a weak-signal edge case but does not justify treating reward
  underflow as the main cause of the modest held-out result.
- **Timing-axis preparation:** the pinned native Qwen processor drops
  punctuation when constructing alignment units and decodes timestamp
  classes in 80 ms increments by default. A timing-only comparison must
  preserve the original ASR text, quote occurrence and word-index anchors,
  mapping aligned units back without fuzzy text replacement. The processor
  already repairs non-monotonic timestamps; validate bounds and retain
  explicit failures instead of dropping questions. These API findings are
  preparation, not evidence that Qwen timing is better.
- Implemented a model-free timing-transfer helper that requires exact
  normalized character order, preserves original ASR text/word indices,
  handles split/merged units and punctuation boundaries, and rejects changed
  doses/negations or invalid times. `check_alignment_tokens.py` checks the
  pinned processor's real tokenization on all 39 frozen transcripts on CPU.
  Its artificial monotonic timestamps exercise mapping only: they must not
  be scored, mistaken for forced-aligner predictions, or used in serving.
- **CPU mapping outcome:** `alignment-token-map-29b40a5d` passed all **39/39**
  frozen transcripts with unchanged original word text/order. The pinned
  processor reports 0.08-second timestamp classes. No neural alignment
  predictions, audio timing quality, or latency gains have been measured.
  Keep the future GPU timing trial separate from the active GRPO OOF run.
- **Real timing probe implemented, not yet run:** `benchmark_alignment.py`
  loads the pinned native Qwen model entirely offline, aligns the original
  MP3 waveform and frozen transcript, and transfers only timestamps.
  Baseline quotes, their exact word-index occurrences, and all yes/no
  decisions remain fixed. Audio/transcript hashes and every baseline
  question/reference/anchor are checked before model loading.
- The predeclared smoke covers the three longest conversations, selected
  without labels. Compare raw Qwen times (no Whisper-fitted correction)
  against the incumbent's +0.2s spans. Invalid proposals or conversation
  errors explicitly retain incumbent evidence and remain in every score
  denominator. Save raw alignment units before checking bounds.
  Profile added time and model-only memory; a separate-run latency sum is
  not a co-resident HTTP gate. Do not launch while GRPO OOF holds the sole
  experimental GPU.
- **Resume/resource constraint:** two serving allocations and another
  medical GPU experiment were active when work resumed on September 19.
  Leave those jobs untouched and keep the existing GPU-capacity guard.
  Continue the same longest-three alignment-quality smoke on CPU instead,
  using an isolated 8-core / 32 GB allocation and the verified interpreter.
- `benchmark_alignment.py --device cpu` loads float32 and explicitly marks
  the result quality-only. It records allocated PyTorch threads, actual
  dtype, CPU time and peak process RSS. Its GPU feasibility flag stays
  false regardless of CPU speed. CPU float32 and later GPU BF16 predictions
  may differ; numerical parity and co-resident HTTP acceptance are still
  required before any serving claim. No serving environment is upgraded.
- **CPU smoke outcome:** `alignment-cpu-smoke-88799922`, all 30 questions
  from the three longest clips, 17 positives. Actual CPU float32 / 8
  threads, no GPU allocation, peak process RSS **10.53 GB**. No
  conversation or span failures; decisions, quotes, word-index anchors
  and references are pointwise unchanged.
  Raw aligned composite **0.6870**, mIoU **0.4783**, versus matched
  incumbent **0.6978**, mIoU **0.4964**. Delta **-0.010826**, interval
  **[-0.024222, +0.003243]**. This is not an improvement.
- Added CPU time was 17.89-18.79 s; maximum separate-run combined estimate
  41.89 s. The GPU feasibility flag correctly remains false.
  Complete one unchanged full-corpus CPU quality comparison: the
  longest-three selection was a stress smoke, not a representative quality
  sample. Preserve this negative subset result; no offset fitting or
  longest-clip exclusion based on its labels.
- **Full unbounded CPU outcome:** `alignment-cpu-full-7d939813` scored all
  390 questions, including explicit incumbent fallbacks. Composite
  **0.7858**, mIoU **0.6447**, versus 0.8007 / 0.6696. On the same 350
  non-demo questions: 0.7828 versus 0.7978, delta **-0.015050**,
  interval **[-0.021038, -0.009380]**. The 30 smoke predictions replayed
  exactly and every decision/quote/anchor/reference was preserved.
- The process exited nonzero because three conversations had a final
  predicted word ending beyond the audio: 0.16s, 0.24s, and 0.16s
  overshoot. Fourteen positive spans consequently kept incumbent evidence.
  Raw units were retained, so this is a diagnosed model-bound failure,
  not missing data or a hidden denominator change.
- Fix decoding at its source: before argmax, restrict timestamp labels to
  those at or before the actual waveform end. This uses audio duration,
  never annotations, and retains the existing strict word/time validation.
  Do not special-case those conversations or silently clamp all timestamps.
  Repeat the same full CPU comparison once with bounded decoding; retain
  the unfavorable unbounded result and make no score-improvement claim.
- **Bounded-decoding outcome:** `alignment-cpu-bounded-403abcbb` completed
  with **zero conversation failures**, all 194 yes spans retimed, all
  39 word streams finite/in bounds, and all 390 decisions/quotes/anchors
  preserved. Raw aligned composite **0.7838**, mIoU **0.6413**; matched
  incumbent 0.8007 / 0.6696. Non-demo score **0.7807**, delta
  **-0.017114**, interval **[-0.023098, -0.011331]**. The bounds defect
  is fixed; the raw timing policy is still worse and is not deployed.
- **Fair calibration diagnostic:** compare independently calibrated
  aligner times with the already-calibrated incumbent. Use exactly the
  original 64-pair offset grid and five conversation folds under seeds
  13 and 37; never tune that grid on this result. Fit only valid aligned
  training spans, exclude every demo-source conversation, and leave
  incumbent fallback spans unchanged rather than correcting them twice.
  Score all 350 non-demo questions, including missed positives and
  fallbacks. `calibrate_alignment.py` writes diagnostic OOF results only,
  not an artifact compatible with the current serving calibration loader.
  CPU float32 results still require GPU parity and HTTP acceptance.
- **Calibration outcome:** `alignment-offset-cv-94bb821e` used all 350
  non-demo questions under both seeds, with verified fit/held-out/demo
  separation and unchanged fallbacks. Every fold chose start **0.0s**,
  end **-0.2s**. Both calibrated results scored **0.7935**, mIoU **0.6578**,
  versus incumbent **0.7978**, mIoU **0.6650**. Delta **-0.004306**;
  intervals **[-0.009458, +0.000783]** and
  **[-0.009424, +0.000541]**. Calibration improves raw Qwen timing, but
  does not justify replacing the incumbent or adding inference overhead.
  Close this timing-provider trial without deployment.
- **Next low-cost control:** apply the same two previously specified,
  unfitted agreement rules to the completed GRPO OOF proposals, instead
  of the earlier SFT proposals. Keep threshold 0.5 and midpoint weight 0.5
  unchanged; no threshold/model fitting on OOF labels. Verify the chosen
  proposal file against its complete OOF score summary and retain every
  baseline decision. This needs no new model or GPU call.
  Use `evaluate_adapter_agreement.py --proposal-kind grpo`; the default
  still reads the original SFT `adapted_oof.json` files.
- **GRPO agreement outcome:** `grpo-agreement-17b507b0` did not qualify
  either rule. Adapter-on-agreement delta **+0.000184**, interval
  **[-0.008438, +0.009260]**; midpoint delta **-0.000664**, interval
  **[-0.004880, +0.003565]**. Both changed 20 questions. Keep the incumbent.

## Medication transcription and ASR vocabulary audit

The user requested an audit of turbo ASR, medical Whisper fine-tunes and
vocabulary prompting. Current cached transcripts contain spelling
discrepancies against positive question wording: `panadil` / Panodil,
`Activel` / Activelle, and `Ibumedin` / Ibumetin. These are candidate
recognition/orthography problems, not audio-verified error labels.
All nine examined positive questions mentioning the selected medicines
were already answered correctly. Better recognition could still change
citation selection; do not infer a score gain from spelling alone.

The wrapper previously supplied neither initial context nor hotwords.
In the pinned faster-whisper 1.2.1 implementation, our
`condition_on_previous_text=False` resets previous tokens between decoded
windows; `initial_prompt` therefore does not persist through the whole
consultation. `hotwords` is supplied to each window. Relevant primary
implementation: [v1.2.1 transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py).

Optional `WHISPER_INITIAL_PROMPT` and `WHISPER_HOTWORDS` are now supported,
disabled by default. Disabled hints preserve the original unprompted cache
hash and transcript schema. Enabled hints are included in the configuration
hash and transcript metadata, and their combined token budget is checked
instead of allowing silent truncation. Existing serving snapshots are not
changed.

`probe_asr_hints.py` compares unprompted CPU turbo, initial context, and
repeated hotwords with the exact same pinned weights, CPU compute type and
random seed. Use a short fixed vocabulary, no doses, full questions or
answer assertions. Disable transcript cache reads/writes. Record word edits,
number/negation changes and newly introduced vocabulary for review.
The cached GPU transcript is a comparison, **not a human reference**;
this diagnostic reports neither WER nor end-to-end competition score.

Primary model candidates reviewed September 19, 2026:

- [Na0s/Medical-Whisper-Large-v3](https://huggingface.co/Na0s/Medical-Whisper-Large-v3):
  a full large-v3 checkpoint trained on doctor/patient material. The model
  card's WER claims are on its own data, not this competition. Full-v3
  decoding cost, medication/dose errors and word timestamps need matched
  assessment before a swap.
- [CrisperWhisper 2.0 turbo](https://huggingface.co/nyralabs/CrisperWhisper2.0_turbo):
  a timing/verbatim-focused alternative, not a demonstrated medical-domain
  winner here. Its card uses a custom runtime and non-commercial research
  license; Pro hotword boosting is a separate commercial feature. Do not
  confuse that restriction with our current faster-whisper hotwords support
  or assume the checkpoint is a drop-in replacement.
- The older backlog's unspecified PriMock57 **turbo** checkpoint remains
  unverified. Do not turn that note into a claimed available model or result.

- **Vocabulary diagnostic outcome:** `asr-med-vocabulary-2ef438ab`
  completed all nine transcriptions using CPU `int8_float32`, with valid
  word times. Initial context did not correct the examined medication
  spellings. Hotwords changed `panadil` to `Panodil` in sample_5, but
  left `Activel` unchanged. CPU control already differed from cached GPU
  output, so those results cannot be transferred silently to GPU.
- Hotwords omitted the repeated clause "spread around the body rather than
  in one place" at 80.88-84.04s in sample_20, present in both unprompted
  transcripts. Other wording still expressed widespread pain, and this
  omission overlapped no supplied positive reference span. It is a
  fidelity warning, not proof of a composite-score regression.
  Negation tokens stayed unchanged; initial context rendered the digit
  `8` as `eight`, a formatting difference rather than a detected dose error.
  No glossary drug was introduced in the sample_4 comparison.
- Keep hints disabled in the qualified service. Next compare the verified
  medical checkpoint with generic **full large-v3**, not only turbo, to
  avoid attributing decoder-size differences to medical adaptation.
  Pin `Na0s/Medical-Whisper-Large-v3` revision
  `9943ad3338e2ffdcdadb193d9e2abc9feeded448` (public, model-card Apache-2.0).
  Its two safe-tensor shards total about 6.17GB; cache the tokenizer's
  `merges.txt` as well, but not `training_args.bin`. Checkpoint metadata
  includes alignment heads, whose conversion must be verified before
  claiming usable word timestamps. Generic full-v3 CT2 control is already
  cached at `edaa852ec7e145841d8ffdb056a99866b5f0a478`.
- **Medical checkpoint cache complete:** `cache-medical-whisper-80ae70dc`,
  12 required files / 6,176,131,893 bytes, exact pinned revision; no
  training pickle or hosted inference. Conversion is run-local and
  offline. `prepare_medical_whisper.py` refuses existing destinations,
  exports the missing fast-tokenizer JSON, verifies exact alignment-head
  preservation, and requires real word timestamps on a 12-second supplied
  training-audio prefix. This is a format/feasibility gate, not a medical
  accuracy or competition-score result.

## ASR stream lifecycle check

- Hard termination of an ASR worker can bypass Python temporary-file cleanup.
  The working implementation now passes an in-memory MP3 stream through the
  installed faster-whisper `BinaryIO` interface, retaining it until lazy
  segments have been consumed.
- CPU run `audio-stream-ea8199a8`, job `25405058`, proved **bit-identical
  float32 waveforms for all 39 clips** between file and stream decoding.
  This is lifecycle hardening, not a claimed score improvement, and does not
  modify the already-frozen live release.
- The sole decision miss (`sample_82_yes_q01`) already has the relevant
  diagnostic term in ASR text. It suggests a clinical-language/certainty
  interpretation issue rather than justification for a broad ASR replacement.
  No question-specific answer rule was added.
