# AGENTS.md — medical-appointment

Nordic AI Cup 2026 case. One doctor/patient consultation arrives as an MP3 plus
ten yes/no questions; return ten booleans and, for every `true`, the audio span
(start/end seconds) the answer was read from.

Read `../README.md` for the full rules. This file is the working brief.

## The task

- **Input** (`POST /predict`): `audio_base64` (plain base64 MP3, no `data:` prefix),
  `audio_filename`, and all ten `questions` for that conversation.
- **Output**: `answers: list[bool]`, `evidence_start: list[float|null]`,
  `evidence_end: list[float|null]`, all three the same length and in question
  order. A `true` must carry a span; a `false` must carry `null`/`null`.
- Question types: `positive` (answer yes), `hard_negative` (near-miss, answer
  no), `off_topic` (answer no). Hard negatives differ by a dose, drug or single
  word — anything that scores by topical overlap lands at the floor.

## Score

`score = 0.4 × accuracy + 0.6 × mean tIoU`

- `mean tIoU` is averaged over **every annotated yes question**, whether or not
  you answered it — a missed positive costs on both halves.
- Point at the passage, not the conversation: a whole-clip span scores ~0.

## Hard constraints

- **No cloud API in the request path.** Everything in `/predict` runs locally.
  Hosted models are fine while developing, never at evaluation time.
- **60 s per request, and 60 s per conversation averaged over the whole attempt.**
  Being slow early takes marks off late conversations.
- **Five consecutive timeouts ends the attempt** — the remaining conversations
  are scored wrong. Any reply (even a 500) resets the counter.
- **Never let `predict` raise.** An exception means no response, which scores
  all ten of that conversation's questions wrong. Catch, log, and return a guess.
- **`api.py` and `dtos.py` are the wire protocol** — leave them alone. The model
  goes in `example.py`.

## Environment

- Conda env: `medical` (Python 3.11). Activate with
  `conda activate medical` (or `source ~/miniforge3/bin/activate medical`).
- Base env is Python 3.14 and has no torch/ctranslate2 wheels — always work in
  `medical`.
- Hardware: WSL, **NVIDIA GTX 1650 with 4 GB VRAM**. This is the binding
  constraint on ASR model size; measure before committing to a model.
- Local state under `models/` and `transcripts/` is gitignored.

## Commands

```bash
conda activate medical
pip install -r requirements.txt          # fastapi/uvicorn/pydantic/requests
python api.py                            # serves http://localhost:9054/predict
python local_evaluator.py                # score the 390 supplied questions
python local_evaluator.py --oracle       # ground truth -> expect 1.000
python local_evaluator.py --verbose      # one line per question
```

The shipped baseline in `example.py` answers `true` to everything and points at
nothing — it scores **0.200** (accuracy 0.500, tIoU 0.000). That is the floor,
not a starting point.

## Agent skills

### Issue tracker

Local-markdown tracker under `.scratch/<feature>/` (this is a solo effort).
See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, label string equal to role name.
See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` in this directory.
See `docs/agents/domain.md`.
