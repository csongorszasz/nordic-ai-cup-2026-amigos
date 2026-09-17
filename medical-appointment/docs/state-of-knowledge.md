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
- **Merged decision premise.** Deciding on clause ± 1 lifted positive recall
  **0.749 → 0.944** (large) at a tiny precision cost (ADR-0004).
- **Hard negatives and off-topic.** 0.930 and 0.943 accuracy (large, merged) —
  NLI alone rejects the near-misses; the numeric guard is a no-op.
- **Proposition rewrite.** Decisive: accuracy `0.821` (base NLI, proposition)
  vs `0.692` for the raw question.
- **The endpoint.** Local + cloudflared: all 200 OK, worst 46.6 s of 60 s, zero
  timeouts on a 19-conversation dry run.

## What doesn't (and why)

**Localization.** Chosen span tIoU ~0.36 against a searched-neighbourhood oracle
of 0.66 and a global candidate oracle of **0.884**. The cause is an objective
mismatch, not tuning: NLI entailment is monotone in context while gold spans are
minimal, so every "pick by score" rule drifts to the longest candidate. Five
selection rules and three search breadths all landed within 0.04 (ADR-0003).

Positive recall — previously the other failure — is resolved by the merged
premise (ADR-0004).

## Loss decomposition (T025, mIoU 0.340)

| stage | ceiling | loss | cause |
| --- | --- | --- | --- |
| any word range | 0.894 | — | ASR word times vs annotator boundaries |
| searched neighbourhood | 0.656 | −0.24 | most-entailing clause ≠ annotated clause |
| chosen span | 0.360 | −0.30 | NLI cannot rank by "what a human cites" |
| mean over positives | 0.340 | −0.02 | 11/195 positives answered no |

Headroom is now almost entirely **localization**. Accuracy headroom ≈ 0.02,
localization ≈ 0.33.

## Budgets (GTX 1650, int8)

- VRAM: 1575 MiB / 4096 MiB (turbo + base NLI). `large-v3` fp16 does not fit
  alongside NLI.
- Latency: ASR 16.1 s mean / 21.6 s worst; NLI 19.2 s (decision 10.5,
  localization 8.8). Validation dry run mean 30.1 s, worst 46.6 s.

## Validation correspondence

| config | score |
| --- | --- |
| shipped baseline (floor) | 0.200 |
| single-clause, large (T018) | 0.512 |
| served cheap config, 3 training convs | 0.471 |
| **served cheap config, validation (T023)** | **0.457** |
| **merged decision, large (T025, not yet served)** | **0.579** |

Local dev_eval predicted the service within noise; the harness and data agree.
T025 has not been served/validated yet — the served config predates the merged
premise.

## Open problems / next

- **M3: learned span ranker** (ADR-0003) — the only large remaining gap
  (localization). Also frees NLI latency for the decision.
- **Re-serve + validate** the merged-decision config (expected ~0.53–0.58).
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
