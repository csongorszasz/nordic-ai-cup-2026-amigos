#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=drone
#SBATCH --time=0-12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g&(a100|h100)"
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# GPU job for drone-flyby. Submitted by idun/submit.sh, which passes the command
# to run in RUN_CMD; it runs from drone-flyby/ on IDUN.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs datasets

source idun/runtime.sh

echo "job ${SLURM_JOB_ID} on ${SLURM_NODELIST}, $(date)"
python idun/preflight.py
echo "RUN_CMD: ${RUN_CMD:-<none>}"
echo "------------------------------------------------------------------------"

[ -n "${RUN_CMD:-}" ] || { echo "No RUN_CMD given"; exit 1; }
bash -euo pipefail -c "$RUN_CMD"

echo "------------------------------------------------------------------------"
echo "done $(date)"
