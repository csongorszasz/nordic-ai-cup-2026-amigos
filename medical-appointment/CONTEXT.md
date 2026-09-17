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
