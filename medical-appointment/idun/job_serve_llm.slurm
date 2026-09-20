#!/bin/bash
# Serve the LLM answerer (ASR + LLM) on a GPU node behind a public tunnel, for
# a validation attempt. The tunnel is selected with MEDAPP_TUNNEL:
#   cloudflared (default) -> ephemeral https://<id>.trycloudflare.com URL
#   ngrok                 -> stable https://$NGROK_DOMAIN URL
#
#   bash idun/submit.sh serve --llm
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

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs captured transcripts

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEDAPP_ANSWERER=llm
export MEDAPP_LLM_MODEL="${MEDAPP_LLM_MODEL:-google/gemma-4-e4b-it}"
export MEDAPP_LLM_DTYPE=float16
export MEDAPP_LLM_FEWSHOT=1
export MEDAPP_INFERENCE_WORKER=1
export MEDAPP_REQUIRE_WARMUP=1
unset MEDAPP_SKIP_WARMUP
export WHISPER_MODEL=large-v3-turbo
export WHISPER_COMPUTE_TYPE=int8
export MEDAPP_CAPTURE=1
export MEDAPP_TUNNEL="${MEDAPP_TUNNEL:-cloudflared}"
export NGROK_DOMAIN="${NGROK_DOMAIN:-}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "effective config: model=${MEDAPP_LLM_MODEL} revision=${MEDAPP_LLM_REVISION:-none} prompt=${MEDAPP_LLM_PROMPT:-base} thinking=${MEDAPP_LLM_ENABLE_THINKING:-unset} effort=${MEDAPP_LLM_REASONING_EFFORT:-unset} tokens=${MEDAPP_LLM_MAX_NEW_TOKENS} legacy=${MEDAPP_LLM_LEGACY_SPECIAL_TOKENS:-0} calibration=${MEDAPP_SPAN_CALIBRATION:-none} tunnel=${MEDAPP_TUNNEL}"
# Speaker tagging is an offline/development step; forwarding MEDAPP_LLM_SPEAKERS
# alone does not implement online diarization.

python api.py > "logs/api_serve_${SLURM_JOB_ID}.log" 2>&1 &
API_PID=$!
TUNNEL_PID=""
TUNNEL_LOG="logs/tunnel_${SLURM_JOB_ID}.log"
cleanup() {
    if [ -n "$TUNNEL_PID" ]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for endpoint (model load)..."
for _ in $(seq 1 320); do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "API exited before readiness; inspect its log." >&2
        exit 1
    fi
    curl -fsS --max-time 3 http://localhost:9054/ >/dev/null 2>&1 && break
    sleep 3
done
curl -fsS --max-time 3 http://localhost:9054/ && echo

URL=""
if [ "$MEDAPP_TUNNEL" = "ngrok" ]; then
    if [ -z "$NGROK_DOMAIN" ]; then
        echo "MEDAPP_TUNNEL=ngrok requires NGROK_DOMAIN." >&2
        exit 1
    fi
    URL="https://${NGROK_DOMAIN}"
    "$HOME/.local/bin/ngrok" http --domain="$NGROK_DOMAIN" http://localhost:9054 \
        --log=stdout > "$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    for _ in $(seq 1 40); do
        grep -aq "started tunnel" "$TUNNEL_LOG" 2>/dev/null && break
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            echo "ngrok tunnel exited before readiness." >&2
            tail -n 40 "$TUNNEL_LOG" >&2 || true
            exit 1
        fi
        sleep 2
    done
elif [ "$MEDAPP_TUNNEL" = "cloudflared" ]; then
    "$HOME/.local/bin/cloudflared" tunnel --url http://localhost:9054 \
        --no-autoupdate > "$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!
    for _ in $(seq 1 40); do
        URL=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$TUNNEL_LOG" 2>/dev/null | tail -1 || true)
        [ -n "$URL" ] && break
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            echo "Tunnel exited before readiness." >&2
            exit 1
        fi
        sleep 2
    done
else
    echo "Unknown MEDAPP_TUNNEL='$MEDAPP_TUNNEL' (expected cloudflared or ngrok)." >&2
    exit 1
fi

test -n "$URL"
echo "PUBLIC_URL: ${URL}/predict"
# Quick-tunnel hostnames can be negatively cached by DNS if queried too early.
# Wait for propagation; never treat a fresh NXDOMAIN as a fatal model failure.
code="000"
for attempt in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "${URL}/" 2>/dev/null || true)
    echo "self-check attempt ${attempt}: ${code}"
    [ "$code" = "200" ] && break
    sleep 5
done
[ "$code" = "200" ] || echo "WARNING: public self-check not 200 after grace; holding anyway." >&2

echo "serving; holding until SLURM time limit (${SLURM_JOB_ID})."
echo "URL: ${URL}/predict"
wait -n "$API_PID" "$TUNNEL_PID"
echo "A serving process exited; terminating this allocation's owned processes." >&2
exit 1
