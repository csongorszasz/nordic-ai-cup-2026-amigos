# Reproducing the evaluation build

This is the exact configuration submitted to the Nordic AI Cup 2026
`medical-appointment` evaluation, plus how to stand it up and what it scores.

| | |
| --- | --- |
| Official evaluation | **0.779** |
| Local HTTP acceptance | **0.8007** (accuracy 0.9974, mean tIoU 0.6696) |
| Validation (19 conversations) | **0.770** (deterministic) |
| Accuracy on the training corpus | 389/390 |

## The build

```
ASR (faster-whisper large-v3-turbo, int8, word timestamps)
  -> whole-transcript local LLM (Gemma-4 26B-A4B, base prompt + 3 LOCO-safe demos)
  -> strict-JSON answers + verbatim evidence_quote
  -> alignment back to ASR word times
  -> constant +0.2 s start boundary calibration
```

| component | value |
| --- | --- |
| LLM | `google/gemma-4-26b-a4b-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| Prompt | base (no `MEDAPP_LLM_PROMPT`), 3 LOCO-safe few-shot turns |
| Output | 1024 tokens, legacy special tokens |
| ASR | `WHISPER_MODEL=large-v3-turbo`, `WHISPER_COMPUTE_TYPE=int8` |
| Boundary calibration | `calibration/span_offset_base.json` (start **+0.2 s**, end **0.0 s**) |
| Answerer | `MEDAPP_ANSWERER=llm`, owned worker, warm-up required |
| Hardware | one **exclusive** 80 GB GPU (≈65 GB VRAM, ~18–22 s/conversation) |

The calibration artifact is provenance-bound: it is only accepted when the ASR
config hash (`e75a7f6e`), the model revision, and `variant=base` all match.

## 1. Environment and prerequisites (IDUN)

**IDUN is the NTNU (Norwegian University of Science and Technology) HPC
cluster** this build was run on; the module/conda/SLURM commands below target
it. An equivalent single 80 GB GPU host works too, but the environment lines are
IDUN-specific.

Create the `nordic` environment (also pre-downloads some models):

```bash
bash idun/setup_env.sh          # run on a login node (compute nodes may lack internet)
```

The serving path then runs offline (`HF_HUB_OFFLINE=1`), so **all weights must
be cached in advance**. `setup_env.sh` alone is not enough for this build —
you also need:

1. **Gemma-4 26B-A4B weights (pinned revision, ~49 GB):**
   ```bash
   python idun/cache_model.py \
     --model google/gemma-4-26b-a4b-it \
     --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
     --max-bytes 60000000000
   ```
2. **The turbo ASR model.** The served ASR is `large-v3-turbo`; `setup_env.sh`
   only pre-fetches `large-v3`. Fetch turbo once on a login node:
   ```bash
   python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"
   ```
3. **Training transcripts for the few-shot demonstrations.** The prompt includes
   demonstrations drawn from the supplied training conversations, and the
   answerer loads all 39 training transcripts at start-up from `transcripts/`
   (gitignored). Populate that cache by running the pipeline once over the
   training audio:
   ```bash
   python dev_eval.py            # transcribes data/audio/... and caches transcripts/
   ```
   Start-up fails closed (`MEDAPP_REQUIRE_WARMUP=1`) if these are missing.

The NLI, MiniLM and ModernBERT downloads in `setup_env.sh` are only used by the
other answerers; this build uses `MEDAPP_ANSWERER=llm`.

## 2. Sync the code

From the project root:

```bash
REMOTE_DIR='~/nordic-medical-ngrok' bash idun/submit.sh sync
```

## 3. Serve the endpoint (one exclusive 80 GB GPU)

```bash
REMOTE_DIR=~/nordic-medical-ngrok EXCLUSIVE=1 CONSTRAINT=gpu80g MEM=128G TIME=1-00:00:00 \
MEDAPP_LLM_MODEL=google/gemma-4-26b-a4b-it \
MEDAPP_LLM_REVISION=4d7ae4984b7db7de8f8457170b3f1a419ee76d52 \
MEDAPP_LLM_MAX_NEW_TOKENS=1024 MEDAPP_LLM_LEGACY_SPECIAL_TOKENS=1 \
MEDAPP_SPAN_CALIBRATION=calibration/span_offset_base.json \
MEDAPP_TUNNEL=ngrok NGROK_DOMAIN=motor-throttle-viewer.ngrok-free.dev \
bash idun/submit.sh serve
```

`idun/job_serve_llm.slurm` sets `MEDAPP_ANSWERER=llm`,
`MEDAPP_INFERENCE_WORKER=1`, `MEDAPP_REQUIRE_WARMUP=1`, turbo int8 ASR, and
`MEDAPP_CAPTURE=1` (debug captures). `submit.sh serve` forwards the LLM model,
revision, token cap, legacy-token flag, prompt, calibration, and tunnel settings.
The job prints `PUBLIC_URL: <host>/predict`. Use `MEDAPP_TUNNEL=cloudflared` for
an ephemeral `trycloudflare` URL instead of the stable ngrok domain.

Readiness: the job waits for the model to load and self-checks `GET /`; on a
shared node port 9054 can be occupied, hence `EXCLUSIVE=1`.

## 4. Verify before submitting

```bash
curl -s https://<host>/                       # "Your endpoint is running!"
curl -s -o /dev/null -w "%{http_code}\n" https://<host>/predict   # 405 = alive (POST only)
```

Optional liveness watchdog (one GET every 300 s, no inference):

```bash
WATCH_URL=https://<host>/predict sbatch --account=share-ie-idi --export=ALL \
    idun/job_watch_endpoint.slurm
```

## 5. Submit

1. Enter `https://<host>/predict` and the team API key on
   <https://cases.nordicaicup.com>.
2. `QUEUE VALIDATION ATTEMPT` (unlimited) → expect ~0.77.
3. `QUEUE EVALUATION ATTEMPT` — **one attempt only**; validation is 19
   conversations, evaluation is 38, sent one at a time (60 s each).

## Local scoring / verification

- In-process score and span oracles: `python dev_eval.py --diagnostics`.
- Full local HTTP acceptance replay (cached ASR disabled): `python http_benchmark.py`.
- Reproduction of the `+0.2` correction: `python replay_parity.py` (a calibrated
  replay matches the endpoint exactly; a raw replay differs by exactly +0.200).
- Per-question tIoU analysis and the disjoint-cause breakdown:
  `notebooks/tio_losses.ipynb`.
- Full experiment ledger and diagnosis: `docs/state-of-knowledge.md`,
  `docs/tries.md`, `docs/post-mortem.md`.

## Notes

- Captures (`captured/`) and transcripts are debug artifacts. Never train on
  validation or evaluation data.
- The `+0.2 s` start offset is the only boundary correction that helped; every
  other extent/offset change is neutral or worse (see `docs/tries.md`).
- The remaining loss is annotation-convention-limited: of 22 disjoint positives,
  8 are genuine model errors, 11 are valid alternative occurrences, and 3 are
  degenerate gold spans (see `docs/post-mortem.md`).
