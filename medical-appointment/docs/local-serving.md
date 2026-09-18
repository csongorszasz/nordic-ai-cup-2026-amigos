# Local serving runbook

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
