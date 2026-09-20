# CONTEXT.md — medical-appointment

Domain glossary. One term, one meaning; use these words in issues, ADRs, and
code. This is a single-context repo (see `docs/agents/domain.md`).

## The task domain

**Conversation** — one simulated English doctor/patient consultation. A single
MP3, one channel shared by both speakers, roughly 1–3.5 minutes. Identified by
`audio_filename` (`conversation_sample_17.mp3`) or `transcript_id` (`sample_17`).
The unit of work: one HTTP request carries exactly one conversation.

**Question** — an English yes/no question about a conversation. Every
conversation is asked exactly ten. Three kinds, and the distinction is the whole
challenge:

- **positive** — a statement the conversation establishes. The true answer is
  *yes*, and it has an **evidence span**.
- **hard_negative** — a near-miss on something the conversation establishes
  (same drug at a different dose, same symptom in a different place). True
  answer *no*, no evidence span. Lexically almost identical to its positive
  counterpart, so topical overlap answers them wrong.
- **off_topic** — a subject that never comes up. True answer *no*, no span.

**Transcript** — the timestampled output of local ASR over a conversation:
a list of **segments**. We never call a hosted ASR service.

**Segment** — one ASR unit with `start`, `end` (seconds) and `text`. Segment
timings are what evidence spans are built from, so they are part of the answer,
not just an intermediate.

**Window** — a contiguous run of words with an exact span, built from the word
timestamps by splitting on sentence punctuation, pauses and segment bounds. It
is the unit the verifier scores and the decision context for localization.
*Phrase-level*, i.e. clause-sized, not segment-sized.

**Proposition** — a question rewritten as a declarative statement
(`Should the daily dose be 100 mg?` → `The daily dose is 100 mg.`). The
hypothesis an NLI verifier tests against a window.

**Sub-range localization** — the legacy step that chooses the returned evidence span:
the tightest contiguous word range inside the winning window ± one neighbouring
window. The span is *not* the window itself (ADR-0001).

**Source unit** — an experimental citation candidate identifying one occurrence
of a clause, sentence, or contiguous multi-sentence episode. Short confirmations
remain separate units. A source unit is not yet a verified supporting passage;
the question determines whether that occurrence and extent are appropriate
(ADR-0005).

**Interpretation context** — surrounding transcript text attached to a source
unit so a short reply or result can be understood. Its words are not automatically
included in the returned evidence span. The current source-unit probe does not
replace the qualified quote-citing service.

**Evidence span** — the `[start, end]` interval (seconds from the start of the
audio) that supports a *yes* answer. A *no* answer has no span: both timestamps
are `null`. Also called a **prediction** once returned.

**Annotated span** — the ground-truth evidence span from `question_train.csv`.
Only the 195 `positive` rows have one.

## Scoring

**Accuracy** — fraction of the ten booleans that match the label.

**tIoU (temporal IoU)** — overlap of predicted and annotated spans divided by
the stretch they cover together. Identical spans score 1, non-touching spans 0.
A whole-conversation span scores near 0.

**mean tIoU** — tIoU averaged over *every annotated positive question*, whether
or not it was answered yes. A missed positive scores 0 there and is also wrong
on accuracy.

**Score** — `0.4 × accuracy + 0.6 × mean tIoU`.

**Floor** — what the shipped baseline (`example.py`: always yes, no spans)
scores: `0.200`.

## Boundaries

**Request** — one `POST /predict` carrying a whole conversation and its ten
questions. Budget 60 s; also 60 s per conversation averaged over an attempt.

**Attempt** — one full run of a dataset through the endpoint. Validation is 19
conversations; evaluation is 38 and is the one scored run.

**5-timeout abort** — five consecutive timed-out requests and the service stops
sending; the unsent conversations are scored wrong. Only timeouts accumulate; a
malformed reply still proves the endpoint is alive and resets the counter.

## Serving

**Serving config** — the environment-variable set that runs the endpoint
(`WHISPER_MODEL`, `NLI_MODEL`, `MEDAPP_TOP_CLAUSES`, …). See ADR-0002.

**Capture** — opt-in recording (`MEDAPP_CAPTURE=1`) of each request's audio and
questions plus our response, for debugging and consistency checks only. Never a
training source.

**Positive recall** — the share of the true positives we answer yes to. A missed
positive is doubly expensive: a wrong answer (accuracy) and a zero in the
`mean tIoU` average, which is fixed over all annotated positives.
