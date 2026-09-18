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

