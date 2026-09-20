#!/bin/bash
# GET-only liveness watchdog for the public /predict endpoint.
#
#   WATCH_URL=https://<host>/predict sbatch --account=share-ie-idi \
#       --export=ALL idun/job_watch_endpoint.slurm
#
# Cheap: one GET per interval (well under ngrok's request limits) and no
# inference, so it never disturbs validation. It logs OK/FAIL to the job output
# and stays up for the allocation's walltime. A dead agent or drained job shows
# as a non-200 / ERR line; a 200 only proves the tunnel reaches the API root.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_watch
#SBATCH --time=2-00:00:00
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --nodes=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

URL="${WATCH_URL:?WATCH_URL must be set}"
INTERVAL="${WATCH_INTERVAL:-300}"
ROOT="${URL%/predict}/"

echo "watchdog ${SLURM_JOB_ID} on ${SLURM_NODELIST} $(date) -> ${ROOT} every ${INTERVAL}s"
fails=0
while true; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
        -H 'Accept: application/json' -H 'User-Agent: medapp-watchdog/1.0' \
        "$ROOT" 2>/dev/null || echo ERR)
    if [ "$code" = "200" ]; then
        fails=0
        echo "$(date -Is) OK 200"
    else
        fails=$((fails + 1))
        echo "$(date -Is) FAIL ${code} consecutive=${fails}" >&2
        echo "$(date -Is) FAIL ${code} consecutive=${fails}"
    fi
    sleep "$INTERVAL"
done
