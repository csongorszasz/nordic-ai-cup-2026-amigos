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

- **Dev/test env:** `medical` (Python 3.11) — fastapi/pydantic/requests/pytest.
  Activate with `conda activate medical`. Run the fast tests here.
- **Serving env:** `medapp-local` (Python 3.11) — torch cu121, faster-whisper,
  transformers. Created by `local/setup_env.sh`. Serves the endpoint locally
  (ADR-0002).
- Base env is Python 3.14 and has no torch/ctranslate2 wheels.
- Hardware: WSL, **NVIDIA GTX 1650 with 4 GB VRAM**. Binding constraint: only
  int8 ASR fits alongside the NLI model. Dev/experiments run on **IDUN** GPUs.
- Local state under `models/`, `transcripts/`, `logs/`, `results/`, `captured/`
  is gitignored.

## Commands

```bash
# dev / scoring (medical env)
conda activate medical
python dev_eval.py --diagnostics          # in-process score + span oracles
python dev_eval.py --answer nli --limit 3 # real pipeline on 3 conversations
python -m pytest -m "not slow"            # fast test gate
python local_evaluator.py --oracle        # harness sanity -> 1.000

# IDUN (see docs/local-serving.md for the full runbook)
bash idun/submit.sh dev --diagnostics     # GPU dev_eval (fast-test gate first)
bash idun/submit.sh test                  # slow (model-backed) tests
bash idun/submit.sh pull                  # copy results/ + transcripts/ back

# local serving
bash local/setup_env.sh
bash local/bench_asr.py                   # ASR speed/VRAM benchmark
python api.py                             # serves http://localhost:9054/predict

# knowledge
docs/state-of-knowledge.md   docs/tries.md   docs/adr/   docs/local-serving.md
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
