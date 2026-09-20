# ADR-0004: Decide on a merged clause window, not a single clause

- **Status:** Accepted
- **Date:** 2026-09-17
- **Context:** medical-appointment, Nordic AI Cup 2026
- **Related:** ADR-0001 (clause windows), ADR-0003 (localization)

## Context

The decision stage scored each clause independently and answered yes when the
best clause entailment cleared τ. Positive recall plateaued at ~0.73 (base) /
~0.75 (large), and threshold tuning could not improve it without a matching loss
on hard negatives.

`diagnose_recall.py` scored the proposition against three premises over the 195
annotated positives:

| premise | base ≥τ | large ≥τ |
| --- | --- | --- |
| best single clause | 73% | 75% |
| **clause ± 1 merged** | **90%** | **95%** |
| gold evidence text | 62% | 66% |

Hard-negative false positives rose from 9% → 13% (base) and 5% → 7% (large) with
the merged premise — a small cost against a large recall gain on a balanced set.

The cause is boundary alignment: **71% of annotated gold spans cross our clause
boundaries**, so a single clause frequently contains only part of the evidence
and cannot entail the claim. (The gold-text score is not a fair ceiling because
annotations are minimal, often truncated fragments; they under-entail without
surrounding context.)

## Decision

Score the decision against **each clause merged with one neighbour on each
side** (`MEDAPP_DECISION_NEIGHBOURS=1`), and use the best such window both for
the yes/no decision and as the anchor for localization.

## Consequences

- Positive recall **0.749 → 0.944**, accuracy **0.851 → 0.938**, mIoU
  **0.286 → 0.340**, score **0.512 → 0.579** (large NLI, T025).
- Cost is negligible: the number of NLI pairs is unchanged; only the premise
  text grows.
- Precision is slightly reduced (hard negatives crossing τ 5% → 7% at large);
  the numeric guard stays as cheap insurance.
- Localization remains the bottleneck (chosen 0.36 vs neighbourhood oracle 0.66)
  and is addressed separately (ADR-0003).
