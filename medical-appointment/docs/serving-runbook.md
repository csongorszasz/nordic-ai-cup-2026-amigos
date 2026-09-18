# Serving runbook — LLM answerer

How to measure, serve and hand in the LLM answerer (validated 0.744, T039)
without losing the one-shot evaluation to an operational failure. Written for
whoever runs the next validation or the evaluation.

## What protects a request

| Risk | Guard | Knob |
| --- | --- | --- |
| Reply cut off by the token limit → all ten "no" | `max_new_tokens` 1024; every answer object that closed is salvaged, only the missing ones are guessed | `MEDAPP_LLM_MAX_NEW_TOKENS` |
| Generation runs past the budget | `generate(max_time=…)` stops at the soft deadline minus a reserve; the partial reply is salvaged | `MEDAPP_DEADLINE_S` (50), `MEDAPP_LLM_RESERVE_S` (2) |
| Anything hangs (ASR, driver, lock) | `predict` waits at most the hard timeout, then answers with guesses and abandons the worker | `MEDAPP_HARD_TIMEOUT_S` (55) |
| An overrun eats the next request's time | one conversation on the GPU at a time; a request waits for the lock only within its own budget, then guesses | — |
| Long audio | ASR stops at the deadline and answers from the partial transcript | — |
| Constant fallback (0.5 accuracy) | question-text prior (`prior.py`, 0.661 held-out conversations, off-topic 1.000) | — |
| Silent zero-shot (no `transcripts/`) | `MEDAPP_STRICT_STARTUP=1` makes warm-up raise; the serve job exits instead of serving | `MEDAPP_STRICT_STARTUP` |
| Model failed to load, `/` still says "running" | `GET /ready` is 200 only after warm-up; the job and the rehearsal wait for it | — |
| Different weights/packages at evaluation | `setup_llm.sh` records revisions (`models/llm_revisions.txt`) and `idun/serving-env.lock.txt`; the job pins the revision | `MEDAPP_LLM_REVISION` |
| API or tunnel dies mid-attempt | the serve job's watchdog restarts either within 30 s | — |
| Tunnel URL changes on restart | named tunnel when `~/.cloudflared/medapp_token` exists on IDUN | `CF_PUBLIC_URL` |

## Measure before switching anything on

Everything that changes the prompt or the spans is **off by default**, so the
served behaviour is still T039's until a probe run says otherwise. The one
exception is the fuzzy quote fallback (`MEDAPP_ALIGN_FUZZY=0.85`): it only
touches yes answers whose quote was not found, which score 0 without it. Run each as
the `SERVED` probe rung, which uses the served answerer itself:

```bash
M="--model google/gemma-4-e4b-it --rungs SERVED"   # the probe defaults to Qwen
bash idun/submit.sh llm $M --tag base                          # baseline, same as T038
bash idun/submit.sh llm $M --tag fs411 --fewshot-counts 4,1,1  # more yes examples
bash idun/submit.sh llm $M --tag seg --cite-segment            # occurrence choice
bash idun/submit.sh llm $M --tag seg_fs --cite-segment --fewshot-counts 4,1,1
bash idun/submit.sh pull
python calibrate_spans.py results/llm_base_SERVED_questions.json          # boundary offsets
```

Adopt a knob only if its in-sample score beats `base`. For offsets, trust the
leave-one-conversation-out number `calibrate_spans.py` prints, not the fitted
one. Then confirm the winning combination with one validation run before the
evaluation.

ASR hotwords (`MEDAPP_ASR_HOTWORDS=1`) change the transcript, so the probe's
cached transcripts cannot measure them: A/B with two validation runs
(`SERVE_EXPORT=MEDAPP_ASR_HOTWORDS=1`). Numbers and units are never hotwords,
because hard negatives differ from the truth by exactly those.

## Serve

```bash
bash idun/setup_llm.sh [--with-26b]    # once, on the login node; commit the lock file
bash idun/submit.sh serve                                         # E4B, 4 h
SERVE_MODEL=26b SERVE_TIME=0-12:00:00 bash idun/submit.sh serve   # 26B-A4B (0.755 in-sample)
SERVE_EXPORT="MEDAPP_LLM_CITE_SEGMENT=1,MEDAPP_LLM_FEWSHOT_COUNTS=4:1:1" bash idun/submit.sh serve  # colons: --export splits on commas
```

The job prints `PUBLIC_URL:` once `/ready` is 200, and writes the effective
configuration to `logs/serve_config_<job>.txt` and the environment to
`logs/serve_env_<job>.txt`.

## Before the evaluation

1. The job has at least 2 h of walltime left (38 conversations × ~20 s is
   ~13 min, but leave room for a restart).
2. `curl <url>/ready` is 200 and shows the intended model, revision, few-shot
   count and knobs.
3. `bash local/rehearse.sh <url>/predict` prints `PASS`: 0 timeouts, 0 failed,
   worst round trip under 45 s, through the **public** URL.
4. `grep -E "guessing|overran|unanswered" logs/api_serve_<job>.log` is empty
   for the rehearsal. Those are 200 responses the harness scores as answers.
5. A human submits the evaluation. One attempt only.
