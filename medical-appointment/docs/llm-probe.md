# LLM ceiling probe

Establish an upper bound on the task with a local instruction model: one call per
conversation, transcript-only (no retrieved candidates), strict JSON with a
verbatim `evidence_quote` aligned back to ASR word timestamps. The score is
**in-sample/ceiling** over the 39 supplied conversations, comparable to the
ModernBERT OOF 0.610 and the hybrid 0.647-0.661. It is not a validation number.

## Rungs

| rung | calls | input | purpose |
| --- | --- | --- | --- |
| L0 | 1 | system + timestamped transcript + q01–q10 + schema | raw capability |
| L1 | 1 | L0 + 2–3 LOCO-safe few-shot turns | format + quoting convention |
| L2 | 2 | decide pass, then cite pass for the yes questions | cleaner quoting |

Inputs never include `question_type` (unavailable at evaluation time) and never
include retrieved candidates. Few-shot examples are drawn only from other
conversations so a target's answer cannot leak into its own prompt.

## Run

```bash
# once, on the IDUN login node
bash idun/setup_llm.sh

# smoke, then full
bash idun/submit.sh llm --rungs L0 L1 L2 --limit 3
bash idun/submit.sh llm --rungs L0 L1 L2
```

Local prompt/parse/alignment checks need no GPU:

```bash
conda activate medical
python -m pytest -m "not slow" -q
```

## Outputs

`results/llm_<tag>_<rung>.json` (summary) and
`results/llm_<tag>_<rung>_questions.json` (per-question record: prediction,
span, gold, quote). Metrics: score, accuracy, mIoU, by-type, positive recall,
**quote-found rate** (hallucination proxy), tIoU when answered yes, parse
failures, latency.

## Model

`Qwen/Qwen2.5-7B-Instruct`, fp16, single GPU, greedy decoding, via plain
`transformers` in the `nordic` env (no vLLM for a sequential probe). Override
with `MEDAPP_LLM_MODEL`, `MEDAPP_LLM_MAX_NEW_TOKENS`, `MEDAPP_LLM_DEVICE`,
`MEDAPP_LLM_DTYPE`.

## Serving

7B cannot be served on the 4 GB GTX 1650. The probe is a ceiling/teacher: if it
wins decisively, options are a 3B/int4 served variant, distillation into
ModernBERT, or (now that compute-node egress and cloudflared are verified) an
IDUN-hosted service behind a tunnel — with the job-lifetime/URL risks that
implies.
