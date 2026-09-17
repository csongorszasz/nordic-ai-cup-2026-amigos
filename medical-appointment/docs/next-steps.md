# Next steps

Standing plan for the medical-appointment case, updated 2026-09-17 after the
first successful served validation (T027, **0.545**). Threads A/B/C below;
execution order is a judgement call, not a dependency.

## State

- Validated score **0.545** (merged decision premise, served locally). Floor
  0.200; single-clause served 0.457.
- `dev_eval` predicts the service exactly (training T024 0.545 == validation
  T027 0.545), so iterate locally.
- Remaining headroom is almost entirely **localization**: chosen span tIoU 0.36
  vs searched-neighbourhood oracle 0.66 vs global candidate oracle 0.88.
- Serving works: local GTX 1650, `large-v3-turbo` int8 + base NLI, cloudflared;
  ~22 s mean / 31 s worst (cached transcripts), 1.6 GB VRAM.

## Thread A — the main path

- **A2 (next): learned span localizer** (ADR-0003). Train on the 195 gold spans,
  LOCO over 39 conversations. Feature ranker (GBM/logreg) first; DeBERTa-base
  `(question, span)` cross-encoder second. Wire into `answer.py` as the selector.
  Side benefit: removes ~1000 candidate NLI calls/conv → frees latency for a
  large-NLI decision. Gate: LOCO mIoU > 0.34 and score > 0.579.
- **Neighbour sweep (optional, cheap):** `MEDAPP_DECISION_NEIGHBOURS ∈ {0,1,2,3}`
  base/large, decision-only, to confirm the optimum sits at ±1 (recall saturates,
  precision keeps falling). ~10–15 min on IDUN.
- **Large NLI for the decision:** +0.03 on training (0.579) but ~2.5× cost; needs
  premise-encoding reuse / batched hypotheses to fit the 60 s budget on the 1650.
- **Serving:** local + **named tunnel** (stable URL) requires a Cloudflare-managed
  domain; quick tunnel meanwhile. Re-serve + validate after A2.

## Thread B — LLM method (ceiling probe)

Can a local instruction LLM beat NLI on the decision and localize via a verbatim
quote resolved to word timestamps?

- **Where:** IDUN GPUs. 7–8B instruct (Qwen2.5-7B / Llama-3.1-8B) via vLLM or
  llama.cpp.
- **Shape:** one call per conversation — timestamped, id'd transcript + all ten
  questions → strict JSON `{answer, evidence_quote}`; few-shot examples from the
  training split.
- **Scoring:** reuse `dev_eval` metrics; align the quote back to `Word.start/end`
  for tIoU. Compare against 0.545 (base) / 0.579 (large).
- **Serving:** 4 GB cannot run 7B usefully → likely a ceiling probe, or a 3B/int4
  / GGUF-on-CPU variant if it wins.
- **Risks:** hallucinated/unfindable quotes, latency, prompt sensitivity; no tuning
  on validation.

## Thread C — medical-domain ASR

Improve transcription of medical terms (drugs, doses) and see if it moves the
score. Research done:

- **Candidate model:** `Na0s/Medical-Whisper-Large-v3` — whisper-large-v3
  fine-tuned on PriMock-derived doctor/patient data; self-reported WER 0.19 vs
  0.33 baseline. Also `xpoon/medical-whisper-large-v3-ggml` (Q5_0 ≈ 1.0 GB) and
  `Knowtex-ai/whisper-medical-govcloud`.
- **Data:** **PriMock57** — 57 mock primary-care consultations with audio +
  utterance-level transcripts + notes (`github.com/babylonhealth/primock57`;
  text form on HF). Built as a medical-ASR benchmark.
- **Phase 1 (running):** WER on a PriMock subset for large-v3 / turbo /
  Medical-Whisper-Large-v3; convert the medical HF model to CTranslate2 for
  `faster-whisper`. Transcribe on IDUN.
- **Phase 2 (gate):** transcribe our 39 with the medical model (separate cache
  hash), run `dev_eval`, compare to 0.545. Pursue only if the downstream score
  improves — our ASR already preserves doses.
- **Phase 3 (optional):** fine-tune **turbo** on PriMock57 on IDUN, convert and
  re-evaluate. Serving a fine-tuned large-v3 int8 is ~3.6× on the 1650 (too slow
  alongside NLI), so turbo is the serving-friendly target.
- **Constraints:** no ground-truth transcripts for our 39 (only WER on PriMock +
  downstream score); check PriMock/model licences; audio size.

## Hygiene / open questions

- Captures (`captured/`) and transcripts (`transcripts/`) are debug artifacts —
  never train on validation/evaluation data.
- Open: Cloudflare domain for the named tunnel? Priority between A2, B and C?
  Whether to fine-tune (Thread C Phase 3) or just use the existing medical model.
