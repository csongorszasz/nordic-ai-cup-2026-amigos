# Local serving runbook

## Current IDUN experiment path

The approved main target is IDUN; the GTX 1650 instructions below describe the
older local fallback. From PowerShell, use `python idun\run.py` to create an
isolated snapshot rather than synchronizing over the active service:

```powershell
python idun\run.py submit --tag compare --gpu80 --script benchmark.py `
  --env WHISPER_MODEL=large-v3-turbo --env WHISPER_COMPUTE_TYPE=int8 -- `
  --model google/gemma-4-26b-a4b-it `
  --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --variants base v2 --force-asr
python idun\run.py status --run-id <returned-run-id>
python idun\run.py wait --run-id <returned-run-id>
python idun\run.py pull --run-id <returned-run-id>
```

The helper snapshots source and request configuration, copies only training
transcripts, preserves remote-only source/captures, and prevents overlapping
experimental jobs. Summaries and logs are pulled under `results\idun_runs`.
An 80 GB allocation is necessary for the current fp16 26B comparison.

For a **qualified candidate** server, `MEDAPP_INFERENCE_WORKER=1` isolates ASR
and LLM in a preloaded owned process. `MEDAPP_REQUIRE_WARMUP=1` refuses startup
when model/demonstration preparation fails. The parent enforces
`MEDAPP_DEADLINE_S` (default 50 seconds), returns valid no/null guesses on a
stalled worker, and reloads it without queueing requests behind the reload.
This mode is opt-in until its uncached HTTP gate passes. The active incumbent
has not been changed.

All inference artifacts must be available locally. Pin `MEDAPP_LLM_REVISION`
and retain the run's package/GPU/transcript metadata. New local results do not
authorize competition validation or the one-shot evaluation submission.

Use `--reference-run <run-id>` when staging a follow-up: it copies the prior
isolated run's training audio/transcripts rather than reading mutable live
caches. Explicit small JSON model artifacts can be packaged with
`--asset models\span_offset_base.json`. `MEDAPP_SPAN_CALIBRATION` enables a
context-checked boundary-calibration artifact; leave it unset for the baseline.
On IDUN, the environment value uses a Linux-relative path such as
`models/span_offset_base.json`.

`http_benchmark.py` runs the final local acceptance replay through a separately
owned loopback socket, with ASR caching disabled and the deadline worker
enabled. It creates no tunnel and does not contact competition services.
Use `--shuffle-seed` to exercise question-order handling.

For a qualified long-running release, submit with `--role serving` and an
explicit `--walltime`, and pass `--serve --min-score <gate>` to
`http_benchmark.py`. It replays the full corpus before exposing a tunnel.
`python idun\run.py ready --run-id <id>` then verifies the public root and
the first supplied conversation against the release's own accepted predictions.
Only that successful external check permits `endpoint.json` publication.
TLS verification stays enabled. New quick-tunnel hostnames can be negatively
cached by DNS if queried too early, so future releases include a propagation
grace period rather than treating a fresh NXDOMAIN as a model failure.

The helper refuses a third medical GPU allocation. CPU-only analyses use
`--cpu` and a separate CPUQ job/lock; they cannot load a visible GPU.
Their default allocation remains two cores / 8 GB. Explicit
`--cpu-cores 8 --cpu-memory-gb 32` overrides those CPU-only resources and
records the allocation arguments in the run manifest. GPU resource rules
are unchanged, and CPU overrides are rejected for GPU jobs.
Serving allocations remain finite, and quick tunnels have no uptime guarantee.
Do not change competition submission settings automatically.

### Isolated metric-reward training

`requirements-grpo.txt` is **not** a serving upgrade. Create its overlay in a
fresh CPU snapshot with `idun\setup_lora_env.py --kind grpo`; it pins TRL,
datasets and a compatible torch/vision/audio stack while leaving the shared
environment untouched. Inspect `results/grpo_environment.json` before use.
The setup's torch/cu126 wheels require a compatible driver; the checked
IDUN A100 allocation reported 575.57.08.

Submit `train_quote_grpo.py` through `idun\run.py` with `--gpu80`,
`MEDAPP_PYTHON=<environment-run>/.grpo-env/bin/python`, and script arguments
`--baseline <frozen-run>/results/benchmark --sft-directory <matching-fold-directory>`.
Use `--smoke` first. A successful smoke is only a feasibility gate; a pilot
requires a fresh run starting again from the original SFT adapter, not the
smoke checkpoint. Fold 0 currently uses the frozen `lora_pilot` directory.
All paths passed to the remote script are IDUN paths.

The script keeps 26B decisions fixed, validates data/split/model provenance,
and compares SFT versus GRPO in the same runtime. It does not register an
answerer, alter the live service or contact competition endpoints. A pilot
gain still needs complete OOF and full uncached serving acceptance.

`train_quote_grpo_oof.py` completes the remaining four folds sequentially
with the same interpreter. Pass `--baseline`, `--pilot-run` (the completed
GRPO pilot snapshot), and `--sft-oof-run` (the existing SFT OOF snapshot).
It preflights every warm-start split and adapter, pins pilot source/runtime
and hyperparameters, and reuses fold 0. Partial outputs are explicitly not
complete OOF; only the final summary certifies all 350 questions.

For independent alignment preparation, run `check_alignment_tokens.py`
through a CPU snapshot with `--baseline <frozen-run>/results/benchmark` and
the verified GRPO interpreter. This loads only the pinned Qwen processor,
not model weights. Its report explicitly labels synthetic timing values;
passing it establishes token/index compatibility, not localization accuracy.

After that CPU preflight and once the experimental GPU is free,
`benchmark_alignment.py --baseline <frozen-run>/results/benchmark --smoke`
runs the real pinned aligner on the three longest clips. Use the verified
GRPO interpreter. It never re-runs the decision LLM or changes quote
occurrences. Remove `--smoke` only after feasibility; full results must
include every question, failure fallback, and the demonstration-disjoint
paired comparison. Zero new offsets are the primary alignment policy.

When GPU capacity is occupied, the same benchmark can run as a **CPU
quality diagnostic**: submit with `--cpu --cpu-cores 8 --cpu-memory-gb 32`
and script argument `--device cpu --smoke`. It uses float32 and bounds
PyTorch threads to `SLURM_CPUS_PER_TASK`. CPU completion never passes the
GPU-feasibility flag; CPU timings and dtype-dependent predictions do not
qualify the intended GPU service.

`calibrate_alignment.py --baseline <baseline>/results/benchmark --aligned
<alignment-results-directory>` provides a model-free CPU diagnostic after
a complete successful bounded alignment run. It uses the existing offset
grid with conversation folds, corrects only valid new aligner spans, and
preserves all incumbent fallbacks. No serving calibration artifact is
written: aligner model/revision/dtype must remain distinct from the
original Whisper timing context.

Serve `/predict` from this box (GTX 1650, WSL) and expose it via cloudflared.
Decision and evidence: ADR-0002. Latency budget: ~35 s mean, worst ~47 s, of the
60 s limit.

## 1. Environment

```bash
bash local/setup_env.sh          # creates the `medapp-local` conda env + weights
```

This installs torch (cu121, sm_75), faster-whisper and transformers, and
pre-downloads `faster-whisper-large-v3`, `*-turbo` and the base NLI model.
`local/bench_asr.py` measures ASR speed/VRAM for any model set.

## 2. Serving configuration

Set via environment (all read at process start):

| variable | value | why |
| --- | --- | --- |
| `WHISPER_MODEL` | `large-v3-turbo` | 7.9× RTF on the 1650; doses preserved |
| `WHISPER_COMPUTE_TYPE` | `int8` | required to coexist with NLI in 4 GB |
| `NLI_MODEL` | base MNLI | fast; large only if latency allows |
| `NLI_DEVICE` | `cuda` (or `cpu` to free VRAM) | see `verifier/nli.py` |
| `MEDAPP_TOP_CLAUSES` | `1` | search breadth |
| `MEDAPP_SELECT` | `greedy_trim` | trim long spans back to the evidence |
| `MEDAPP_MAX_CANDIDATES` | `120` | latency cap |
| `MEDAPP_MAX_RANGE_WORDS` | `10` | latency cap |
| `MEDAPP_NLI_TAU` | `0.3` | decision threshold |
| `MEDAPP_DECISION_NEIGHBOURS` | `1` | merge clause ± 1 for the decision (ADR-0004) |
| `MEDAPP_DEADLINE_S` | `50` | stop answering and guess if a request runs long |
| `MEDAPP_ASR_CACHE` | `1` | set `0` to disable transcript caching on serving |
| `MEDAPP_CAPTURE` | `1` | record requests/responses to `captured/` |
| `MEDAPP_ANSWERER` | `legacy` | `modernbert` uses the trained scorer (see below) |
| `MEDAPP_MB_CHECKPOINT` | — | path to `final.pt` when `MEDAPP_ANSWERER=modernbert` |
| `MEDAPP_MB_TAU` | `0.16` | ModernBERT decision threshold (LOCO-calibrated) |

### ModernBERT answerer

Out-of-fold it beats the legacy pipeline (T033 0.610 vs legacy T030 0.569:
mIoU 0.444 vs 0.316), and on the 1650 the scoring path is ~2.5 s cached
(ASR adds ~16–22 s), ~1 GB VRAM. To serve it:

```bash
export MEDAPP_ANSWERER=modernbert
export MEDAPP_MB_CHECKPOINT=models/modernbert_final/final.pt
export MEDAPP_MB_TAU=0.16
bash local/serve.sh start
```

The legacy path stays the default until the flip is signed off.

## 3. Start / stop

Preferred: the supervisor script (server + tunnel, prints the URL).

```bash
bash local/serve.sh start     # start both, prints https://<id>.trycloudflare.com/predict
bash local/serve.sh url       # re-print the current URL
bash local/serve.sh status    # pids + GPU memory
bash local/serve.sh stop
```

Manual equivalent:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate medapp-local
cd ~/nordic-ai-cup-2026-amigos/medical-appointment
WHISPER_MODEL=large-v3-turbo WHISPER_COMPUTE_TYPE=int8 NLI_DEVICE=cuda \
  MEDAPP_NLI_TAU=0.3 MEDAPP_DECISION_NEIGHBOURS=1 MEDAPP_TOP_CLAUSES=1 \
  MEDAPP_SELECT=greedy_trim MEDAPP_TRIM_MARGIN=0.1 \
  MEDAPP_MAX_CANDIDATES=120 MEDAPP_MAX_RANGE_WORDS=10 \
  MEDAPP_CAPTURE=1 setsid nohup python api.py > /tmp/opencode/api_local.log 2>&1 &
```

Models load at import (~15–25 s) before the endpoint answers.

```bash
curl -s http://localhost:9054/        # "Your endpoint is running!"
pkill -f "[p]ython api.py"            # stop
```

## 4. Tunnel

cloudflared binary: `~/.local/bin/cloudflared` (static, no sudo).

**Quick tunnel (dry runs):**
```bash
setsid nohup ~/.local/bin/cloudflared tunnel --url http://localhost:9054 \
  > /tmp/opencode/cloudflared.log 2>&1 &
grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/opencode/cloudflared.log
```
Submit `<url>/predict`. The URL is **ephemeral** — a restart changes it. With no
Cloudflare domain, this is the fallback for validation and the evaluation; keep
`bash local/serve.sh url` handy and re-submit if it changes.

**Named tunnel (stable URL; use for the evaluation):** requires a domain on a
Cloudflare account.
```bash
cloudflared tunnel login                 # authorize the zone (browser)
cloudflared tunnel create medapp         # writes ~/.cloudflared/<id>.json
cloudflared tunnel route dns medapp med.example.com
cat > ~/.cloudflared/config.yml <<'YAML'
tunnel: medapp
credentials-file: /home/dominic/.cloudflared/<id>.json
ingress:
  - hostname: med.example.com
    service: http://localhost:9054
  - service: http_status:404
YAML
cloudflared tunnel run medapp            # stable: https://med.example.com/predict
```

## 5. Validation / evaluation

1. Enter `https://<host>/predict` + team API key on cases.nordicaicup.com.
2. **Verify** (format), then **Queue validation**.
3. Watch `logs`/`captured`; latency comes from the capture `latency_s` field.

## 6. Capture and hygiene

`MEDAPP_CAPTURE=1` writes `captured/<stem>.json` (questions, answers, spans,
latency) and `captured/audio/<stem>.mp3`. Both are gitignored. Captures are for
**debugging and consistency** only — never train on validation/evaluation data
(evaluation is a different set).

## 7. Troubleshooting

- **CUDA OOM** — lower `MEDAPP_MAX_CANDIDATES`, set `NLI_DEVICE=cpu`, or use a
  smaller ASR (`distil-large-v3`, `medium.en`, `small.en`).
- **Requests timing out** — check `nvidia-smi`; a sleeping GPU or another job on
  the card will blow the budget. Confirm with `local/bench_asr.py`.
- **Tunnel down / URL changed** — restart cloudflared; for the evaluation use the
  named tunnel so the URL is stable.
- **Machine asleep** — WSL/Windows sleep drops both the server and the tunnel.
