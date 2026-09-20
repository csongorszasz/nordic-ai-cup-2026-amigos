#!/bin/bash
# Serve the Gemma 4 E4B L1 answerer (ASR + LLM) on a GPU node behind a
# cloudflared quick tunnel, for a validation attempt.
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

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python api.py > "logs/api_serve_${SLURM_JOB_ID}.log" 2>&1 &
API_PID=$!
CF_PID=""
cleanup() {
    if [ -n "$CF_PID" ]; then
        kill "$CF_PID" 2>/dev/null || true
        wait "$CF_PID" 2>/dev/null || true
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

"$HOME/.local/bin/cloudflared" tunnel --url http://localhost:9054 \
    --no-autoupdate > "logs/tunnel_${SLURM_JOB_ID}.log" 2>&1 &
CF_PID=$!
CF_LOG="logs/tunnel_${SLURM_JOB_ID}.log"

URL=""
for _ in $(seq 1 40); do
    URL=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$CF_LOG" 2>/dev/null | tail -1 || true)
    [ -n "$URL" ] && break
    if ! kill -0 "$CF_PID" 2>/dev/null; then
        echo "Tunnel exited before readiness." >&2
        exit 1
    fi
    sleep 2
done
echo "PUBLIC_URL: ${URL}/predict"
test -n "$URL"
curl -fsS -o /dev/null -w "self-check / -> %{http_code}\n" --max-time 20 "${URL}/"

echo "serving; holding until SLURM time limit (${SLURM_JOB_ID})."
echo "URL: ${URL}/predict"
wait -n "$API_PID" "$CF_PID"
echo "A serving process exited; terminating this allocation's owned processes." >&2
exit 1
