# Evidence annotations

Extended evidence labels for training a learned answerer/localizer. Built from
the supplied training data plus an agent-drafted, NLI-validated annotation pass.

## Files

| file | what |
| --- | --- |
| `drafts/ALL.json`, `drafts/<transcript_id>.json` | raw agent annotation output (provenance) |
| `evidence.csv` | the merged label table — the artifact to train on |
| `nli_check.csv` | entailment of each quote against its question proposition |
| `review.md` | flagged rows for human review (edit the `decision` column) |
| `build.py` | merges drafts + gold spans + OOF flags into `evidence.csv` |
| `nli_check.py` | NLI-validates the quotes |
| `review.py` | builds `review.md` from the flags |

## `evidence.csv` schema

`question_id, transcript_id, question_type, bucket, quote, start, end,
quote_source, confidence, nli_p, flags`

- **bucket**: `support` (true), `refute` (near-miss stated with a different
  value), `absent` (topic/detail never appears).
- **start/end**: seconds. Positives use the supplied gold span; hard negatives
  use the agent quote aligned back to word timestamps; absent rows are blank.
- **quote_source**: `gold` (positives), `agent` (negatives), `n/a` (off-topic).
- **flags**: `low_confidence`, `nli_error` (a hard negative the current NLI gets
  wrong — high value), `nli_miss` (a positive the current NLI misses),
  `alignment_failed`.

## Decisions and QC

- **Positives keep the gold spans only** (`support`); the agent's positive quotes
  are a different-but-valid convention and are not used as labels.
- **Hard negatives** use the agent's refute/absent decision; **125** have a
  refute passage, **17** are absent.
- **Refutes were NLI-validated**: mean entailment of the question proposition
  against the refute quote is **0.018**; only 3/125 exceed τ=0.3 (flagged for
  review). So the quotes genuinely do not entail the question.
- **Supports are often "weak" by NLI** (75/195 < 0.3) — expected, because gold
  spans are minimal fragments (matches the `gold_text` ceiling in `diagnose_recall`).
  The claim is entailed by the span *plus context*, not the bare fragment.
- **NLI cross-reference:** all 10 hard negatives the current NLI wrongly answers
  "yes" now have a refute rationale; all 10 positives it misses have support.

## How it is used

- 3-way labels (`support` / `refute` / `absent`) + rationale spans for a
  candidate-scoring model (ModernBERT): for each `(question, candidate)` pair,
  the training target is the bucket and the candidate's **actual tIoU** against
  the gold (positive) or refute (negative) span — dense supervision.
- Minimal-pair contrastive signal: a refute quote is usually the perturbation of
  a positive question in the same conversation.
- `nli_p` comes from `results/oof_T030.json` (out-of-fold) and is for flagging
  only — never used as a training target.

## Review workflow

1. `python annotations/build.py && python annotations/nli_check.py && python annotations/review.py`
2. Edit `review.md` decisions (`keep` / `change-to-absent` / `change-to-refute`).
3. Apply accepted changes and commit `evidence.csv`.
