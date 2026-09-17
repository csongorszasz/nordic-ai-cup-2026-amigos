# State of knowledge

Consolidated findings for the medical-appointment case, as of 2026-09-17.
Detail lives in `docs/tries.md` (experiments), `docs/adr/` (decisions) and
`docs/local-serving.md` (runbook).

## The task and the score

One conversation (MP3) and ten yes/no questions per request; output ten booleans
plus an evidence span per yes. `score = 0.4·accuracy + 0.6·mean tIoU`, where
`mean tIoU` is averaged over **every annotated yes**, whether or not we answered
it. Evidence is the larger half, and a missed positive costs on both halves.

## Pipeline

`ASR (word timestamps) → clause windows → proposition rewrite → NLI decision →
guarded sub-range localization → answer + span`, with a never-raise contract.

## What works

- **ASR.** `faster-whisper` with word timestamps; `large-v3` and `large-v3-turbo`
  preserve doses/numbers (`one million IU four times daily`,
  `fluconazole 50 milligrams`). Turbo is 2.2× faster than large-v3 on the 1650.
- **Hard negatives and off-topic.** 0.94–0.97 and 0.98 accuracy — the numeric
  guard is currently a no-op, i.e. NLI alone rejects the near-misses at τ 0.3.
- **Proposition rewrite.** Decisive: accuracy `0.821` (base NLI, proposition)
  vs `0.692` for the raw question.
- **The endpoint.** Local + cloudflared: all 200 OK, worst 46.6 s of 60 s, zero
  timeouts on a 19-conversation dry run.

## What doesn't (and why)

1. **Localization.** Chosen span tIoU ~0.25 against a **0.884** pruned-candidate
   oracle. The cause is an objective mismatch, not tuning: NLI entailment is
   monotone in context while gold spans are minimal, so every "pick by score"
   rule drifts to the longest candidate. Five selection rules and three search
   breadths all landed within 0.04 (ADR-0003).
2. **Positive recall.** ~0.73 on training; validation yes-rate 37% on a balanced
   set. A missed positive is both a wrong answer and a 0 in the tIoU average.

## Loss decomposition (T018, mIoU 0.286)

| stage | ceiling | loss | cause |
| --- | --- | --- | --- |
| any word range | 0.894 | — | ASR word times vs annotator boundaries |
| searched neighbourhood | 0.714 | −0.18 | most-entailing clause ≠ annotated clause |
| chosen span | 0.381 | −0.33 | NLI cannot rank by "what a human cites" |
| mean over positives | 0.286 | −0.10 | 49/195 positives answered no |

Headroom: accuracy ≈ 0.06, localization ≈ 0.35. **Localization is where the
points are.**

## Budgets (GTX 1650, int8)

- VRAM: 1575 MiB / 4096 MiB (turbo + base NLI). `large-v3` fp16 does not fit
  alongside NLI.
- Latency: ASR 16.1 s mean / 21.6 s worst; NLI 19.2 s (decision 10.5,
  localization 8.8). Validation dry run mean 30.1 s, worst 46.6 s.

## Validation correspondence

| config | score |
| --- | --- |
| shipped baseline (floor) | 0.200 |
| best training config (T018) | 0.512 |
| served cheap config, 3 training convs | 0.471 |
| **served cheap config, validation** | **0.457** |

Local dev_eval predicted the service within noise; the harness and data agree.

## Open problems / next

- **M3: learned span ranker** (ADR-0003) — attack localization, and free NLI
  latency for a stronger decision.
- **Decision recall** — re-tune τ / consider large NLI for the decision, LOCO on
  training only.
- **Serving** — local + **named tunnel** chosen (ADR-0002); set up the stable URL
  before the evaluation.
- **Hygiene** — captures are debug-only; never train on validation/evaluation.

## Artifact map

| what | where |
| --- | --- |
| pipeline | `asr.py`, `windows.py`, `questions.py`, `guards.py`, `verifier/`, `answer.py`, `example.py` |
| scoring / experiments | `dev_eval.py`, `docs/tries.md` |
| serving | `local/`, `docs/local-serving.md`, `capture.py` |
| IDUN dev | `idun/`, `docs/azure-gpu-quota.md` (why not Azure) |
| decisions | `docs/adr/` |
