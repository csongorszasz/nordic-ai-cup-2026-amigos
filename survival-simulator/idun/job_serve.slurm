#!/bin/bash
# Serve the agent endpoint on a compute node behind a cloudflared quick tunnel.
#
#   bash idun/submit.sh serve configs/controller.json
#   bash idun/submit.sh serve training-results/ppo-123/policy.json
#
# The policy config is loaded once at startup; restart to change it. The
# evaluation runs three attempts back to back, so this job holds until the
# SLURM time limit.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sv_serve
#SBATCH --time=0-12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"

export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
POLICY_CONFIG="${1:-configs/controller.json}"
export SURVIVAL_POLICY_CONFIG="${POLICY_CONFIG}"
export SURVIVAL_SINGLE_STREAM=1

CF="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
if [ ! -x "$CF" ]; then
    echo "installing cloudflared..."
    mkdir -p "$(dirname "$CF")"
    curl -sL --max-time 300 \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -o "$CF" && chmod +x "$CF"
fi

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
echo "policy=${POLICY_CONFIG}"

python agent_server.py > "logs/agent_server_${SLURM_JOB_ID}.log" 2>&1 &
API_PID=$!
cleanup() {
    kill "${API_PID}" 2>/dev/null || true
    pkill -f "cloudflared tunnel --url http://localhost:9052" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for agent server..."
for _ in $(seq 1 60); do
    curl -s --max-time 3 http://localhost:9052/ >/dev/null 2>&1 && break
    sleep 2
done
curl -s --max-time 3 http://localhost:9052/ && echo

setsid nohup "$CF" tunnel --url http://localhost:9052 --no-autoupdate \
    > "logs/tunnel_${SLURM_JOB_ID}.log" 2>&1 &
CF_LOG="logs/tunnel_${SLURM_JOB_ID}.log"

URL=""
for _ in $(seq 1 40); do
    URL=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$CF_LOG" 2>/dev/null | tail -1)
    [ -n "$URL" ] && break
    sleep 2
done
echo "PUBLIC_URL: ${URL:-NONE}"
[ -n "$URL" ] && curl -s -o /dev/null -w "self-check / -> %{http_code}\n" --max-time 20 "${URL}/"

echo "serving; holding until the SLURM time limit (${SLURM_JOB_ID})."
echo "URL: ${URL}"
sleep infinity
