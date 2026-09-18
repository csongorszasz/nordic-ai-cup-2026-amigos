#!/bin/bash
# Serve the LLM answerer (ASR + Gemma) on a GPU node behind a cloudflared
# tunnel, for a validation attempt or the one-shot evaluation.
#
#   bash idun/submit.sh serve                 # Gemma 4 E4B (validated 0.744)
#   SERVE_MODEL=26b bash idun/submit.sh serve # Gemma 4 26B-A4B on an 80 GB node
#   SERVE_TIME=0-12:00:00 bash idun/submit.sh serve
#
# Tunnel: if ~/.cloudflared/medapp_token exists on IDUN (a named tunnel's
# token, `cloudflared tunnel token <name>`), a named tunnel is run and its
# hostname is stable across restarts; set CF_PUBLIC_URL to that hostname.
# Otherwise an ephemeral quick tunnel is used (URL changes if it restarts).
#
# Every serving knob can be overridden with `sbatch --export=ALL,VAR=value`
# (list values with colons, e.g. MEDAPP_LLM_FEWSHOT_COUNTS=4:1:1).
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_serve_llm
#SBATCH --time=0-04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs captured

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEDAPP_ANSWERER="${MEDAPP_ANSWERER:-llm}"
export MEDAPP_LLM_MODEL="${MEDAPP_LLM_MODEL:-google/gemma-4-e4b-it}"
# Pinned by idun/setup_llm.sh (models/llm_revisions.txt); empty = cached snapshot.
if [ -z "${MEDAPP_LLM_REVISION:-}" ] && [ -f models/llm_revisions.txt ]; then
    MEDAPP_LLM_REVISION=$(awk -v m="$MEDAPP_LLM_MODEL" '$1==m {print $2}' models/llm_revisions.txt)
fi
export MEDAPP_LLM_REVISION="${MEDAPP_LLM_REVISION:-}"
[ -z "$MEDAPP_LLM_REVISION" ] && echo "WARNING: ${MEDAPP_LLM_MODEL} not pinned; run idun/setup_llm.sh"
export MEDAPP_LLM_DTYPE="${MEDAPP_LLM_DTYPE:-float16}"
export MEDAPP_LLM_FEWSHOT=1
export MEDAPP_LLM_FEWSHOT_COUNTS="${MEDAPP_LLM_FEWSHOT_COUNTS:-1,1,1}"
export MEDAPP_LLM_CITE_SEGMENT="${MEDAPP_LLM_CITE_SEGMENT:-0}"
export MEDAPP_SPAN_START_OFFSET="${MEDAPP_SPAN_START_OFFSET:-0}"
export MEDAPP_SPAN_END_OFFSET="${MEDAPP_SPAN_END_OFFSET:-0}"
export MEDAPP_ASR_HOTWORDS="${MEDAPP_ASR_HOTWORDS:-0}"
export WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"
export WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"
# Refuse to come up degraded (no few-shot, failed load) rather than serve it.
export MEDAPP_STRICT_STARTUP=1
# Never cache served transcripts: transcripts/ holds the few-shot sources.
export MEDAPP_ASR_CACHE=0
# Record questions + answers for debugging; audio only on request.
export MEDAPP_CAPTURE="${MEDAPP_CAPTURE:-1}"
export MEDAPP_CAPTURE_AUDIO="${MEDAPP_CAPTURE_AUDIO:-0}"

JOB="${SLURM_JOB_ID}"
echo "Job ${JOB} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
env | grep -E "^(MEDAPP_|WHISPER_)" | sort | tee "logs/serve_config_${JOB}.txt"
python -m pip freeze > "logs/serve_env_${JOB}.txt" 2>/dev/null || true

API_LOG="logs/api_serve_${JOB}.log"
CF_LOG="logs/tunnel_${JOB}.log"
TOKEN_FILE="$HOME/.cloudflared/medapp_token"
API_PID=""
CF_PID=""

start_api() {
    python api.py >> "$API_LOG" 2>&1 &
    API_PID=$!
    echo "api pid ${API_PID}; waiting for /ready (model load)..."
    for _ in $(seq 1 300); do
        if ! kill -0 "$API_PID" 2>/dev/null; then
            echo "api exited during warm-up; see ${API_LOG}:"
            tail -n 30 "$API_LOG"
            return 1
        fi
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://localhost:9054/ready)
        [ "$code" = "200" ] && { curl -s http://localhost:9054/ready; echo; return 0; }
        sleep 3
    done
    echo "api not ready after 15 min"
    return 1
}

start_tunnel() {
    if [ -f "$TOKEN_FILE" ]; then
        # Token via the environment, not argv: argv is visible in `ps` on a shared node.
        TUNNEL_TOKEN="$(cat "$TOKEN_FILE")" setsid nohup "$HOME/.local/bin/cloudflared" \
            tunnel --no-autoupdate run >> "$CF_LOG" 2>&1 &
        CF_PID=$!
        URL="${CF_PUBLIC_URL:-}"
        echo "named tunnel pid ${CF_PID}; URL ${URL:-<set CF_PUBLIC_URL>}"
    else
        # Rotate, so the URL grep below cannot pick up a dead tunnel's URL.
        [ -f "$CF_LOG" ] && mv "$CF_LOG" "${CF_LOG}.$(date +%s)"
        setsid nohup "$HOME/.local/bin/cloudflared" tunnel --url http://localhost:9054 \
            --no-autoupdate >> "$CF_LOG" 2>&1 &
        CF_PID=$!
        URL=""
        for _ in $(seq 1 40); do
            URL=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$CF_LOG" 2>/dev/null | tail -1)
            [ -n "$URL" ] && break
            sleep 2
        done
    fi
    echo "PUBLIC_URL: ${URL}/predict"
    [ -n "$URL" ] && sleep 5 && curl -s -o /dev/null -w "self-check /ready -> %{http_code}\n" \
        --max-time 20 "${URL}/ready"
}

cleanup() {
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
    [ -n "$CF_PID" ] && kill "$CF_PID" 2>/dev/null || true
}
trap cleanup EXIT

start_api || exit 1
start_tunnel

# Watchdog instead of `sleep infinity`: a dead API is restarted (and the tunnel
# keeps pointing at the port), a dead tunnel is restarted. A quick tunnel gets a
# NEW URL when restarted — that is logged loudly, since the submitted one is gone.
echo "serving; watchdog every 30 s until the SLURM time limit (${JOB})."
while true; do
    sleep 30
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "$(date +%T) WATCHDOG: api died; restarting."
        start_api || echo "$(date +%T) WATCHDOG: api restart failed"
    fi
    if ! kill -0 "$CF_PID" 2>/dev/null; then
        echo "$(date +%T) WATCHDOG: tunnel died; restarting."
        start_tunnel
        [ -f "$TOKEN_FILE" ] || echo "$(date +%T) WATCHDOG: QUICK TUNNEL URL CHANGED -> ${URL}/predict"
    fi
done
