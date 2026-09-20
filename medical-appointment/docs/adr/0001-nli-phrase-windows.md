# ADR-0001: NLI entailment over phrase windows, with clause-neighbourhood localization

- **Status:** Accepted (provisional — revisit after tuning)
- **Date:** 2026-09-17
- **Context:** medical-appointment, Nordic AI Cup 2026

## Context

One conversation (MP3) and ten yes/no questions per request. Output is ten
booleans plus an evidence span per yes. The score is `0.4*accuracy + 0.6*mean
tIoU`, and `mean tIoU` is averaged over every annotated positive whether or not
we answered it. Two consequences shape the design:

1. Evidence dominates, so spans must be tight — a whole-conversation span
   scores about the same as a wrong one.
2. Hard negatives differ from a true statement by a single dose, drug or entity
   (`100 mg` vs `200 mg`, `after a meal` vs `empty stomach`). Anything that
   scores by topical overlap answers them wrong.

The transcript is local (no cloud ASR/LLM in the request path), so the verifier
must run on our own hardware.

## Decision

1. **ASR:** `faster-whisper` `large-v3` with `word_timestamps=True`,
   `vad_filter=True`, `condition_on_previous_text=False`. Every word keeps
   `start`/`end`; the transcript is cached.
2. **Evidence unit is a phrase-level *window*** (a contiguous run of words with
   an exact span), built by splitting on sentence punctuation, pauses and
   segment bounds, then merging fragments and splitting long spans.
3. **Decision is natural-language inference:** the question is turned into a
   declarative *proposition* (`Should the daily dose be 100 mg?` -> `The daily
   dose is 100 mg`), a cross-encoder (DeBERTa-v3 MNLI) scores entailment of that
   proposition against each candidate window, and yes requires
   `max entailment >= threshold` **and** a numeric/entity guard (if the
   question's asserted number is absent from the window, force no). The guard is
   what defeats hard negatives, where NLI alone is weakest.
4. **Localization is a separate sub-range search:** once a window entails, the
   returned span is the best contiguous *word sub-range* within that window
   expanded by one neighbouring window on each side — **not** the window span
   itself.
5. **Never raise:** `predict` catches per question and returns a well-formed
   guess, because one exception costs all ten marks.

## Evidence

Measured on the 195 annotated positive spans (`dev_eval.py --diagnostics`),
temporal IoU achievable if the span were chosen optimally:

| span source | mean tIoU | >= 0.8 |
| --- | --- | --- |
| clause window span | 0.629 | 55 |
| sub-range inside the clause | 0.794 | 133 |
| **sub-range in clause +/- 1 neighbour** | **0.890** | **174** |
| any word range (upper bound) | 0.894 | 177 |
| clause window + 0.2/0.4 s padding | 0.599 | 39 |

Two facts fall out. First, window spans are the bottleneck for tIoU, so the
returned span must be a word sub-range. Second, a one-clause neighbourhood
already reaches the global ceiling (0.890 vs 0.894), so we do not need global
sliding windows — the candidate set stays bounded. Padding hurts, so do not pad.

## Consequences

- `windows.py` produces clause windows for **decision context**; the returned
  span comes from a separate sub-range search bounded to the best window ± 1.
- The verifier sees one premise per window; localization scores candidate
  sub-ranges, so the per-question NLI budget must be sized for that.
- `dev_eval` reports the "sub-range in clause +/-1" oracle as the target, and
  the word-range oracle as the upper bound, so localization regressions are
  visible without a GPU.
- Revisit if tuning (threshold, window size, NLI model size, retrieval) shows
  clause windows are also limiting recall for the decision itself.
