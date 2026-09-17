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
