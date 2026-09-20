# ADR-0003: Localization is a supervised span ranker, not an NLI score

- **Status:** Accepted (provisional — to be validated by LOCO)
- **Date:** 2026-09-17
- **Context:** medical-appointment, Nordic AI Cup 2026
- **Related:** ADR-0001 (which assumed the span could be found by NLI-scoring sub-ranges)

## Context

ADR-0001 chose an NLI cross-encoder to both decide and localize, with the span
taken from the best-scoring word sub-range in the winning clause ± 1. Trials
T002–T020 falsified the localization half of that:

- For positives answered yes, the chosen span scored **0.37–0.40** while the
  best span inside the *searched* neighbourhood was **0.64–0.73**, and the global
  candidate oracle was **0.884**.
- Five selection rules (argmax long/short, shortest-within-margin, shortest-above-
  threshold, longest-within-margin) and three search breadths all landed within
  **0.04** of each other, all at 0.821 accuracy. Tuning could not move it.
- The example dump showed why: chosen spans were almost always the maximum
  14–16 words, extending earlier and earlier to include context. **NLI entailment
  is monotone in context**, but the gold span is the *minimal* passage a human
  cites. The metric we optimise is anti-correlated with the one we are graded on.
- Swapping to a larger NLI model raised accuracy (0.821 → 0.851) but **not**
  localization (mIoU 0.293 → 0.286). More capacity does not fix a wrong
  objective.

## Decision

- Keep NLI (large) for the **decision** — it is good at its job.
- Make **localization supervised**: a ranker over the candidate spans
  (clause ± 1 neighbourhood) trained on the **195 annotated gold spans**, scored
  on cheap features rather than per-candidate NLI:
  question↔span content overlap, number/unit match, length, position within the
  clause, clause entailment score, neighbour-clause score, ASR word probability.
- Validate **leave-one-conversation-out** (LOCO) over the 39 conversations.
- Start with gradient-boosted trees / logistic regression (robust with 195
  examples, near-zero latency); a `(question, span)` cross-encoder is a later
  capacity upgrade, compared on the same LOCO split.

## Evidence driving the choice

tIoU loss decomposition (T018: mIoU 0.286, by stage):

| stage | ceiling | loss | cause |
| --- | --- | --- | --- |
| any word range | 0.894 | — | ASR word times ≠ annotator boundaries exactly |
| searched neighbourhood (top-3) | 0.714 | −0.18 | most-entailing clause ≠ annotated clause |
| chosen span | 0.381 | −0.33 | NLI cannot rank by "what a human would cite" |
| mean over all positives | 0.286 | −0.10 | 49/195 positives answered no (each scores 0) |

Score = `0.4·acc + 0.6·mIoU`. Accuracy headroom ≈ 0.06; localization headroom
≈ 0.35. Localization is where the remaining points are.

## Consequences

- The ranker also removes ~1000 per-candidate NLI calls per conversation,
  cutting localization NLI from ~8 s to sub-second — a direct serving win
  (ADR-0002) that frees latency for a stronger decision model.
- Gold spans come from the training split only; validation/evaluation captures
  must never be used for training (overfitting risk, and evaluation is a
  different set).
- A positive span requires a candidate set that contains the gold span (the
  pruned-candidate oracle is 0.884), so window construction stays important.
