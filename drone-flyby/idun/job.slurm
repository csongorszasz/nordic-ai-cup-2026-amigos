#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=drone
#SBATCH --time=0-12:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu32g|gpu40g|gpu80g"
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

module purge
module load Anaconda3/2023.09-0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate drone

export PYTHONUNBUFFERED=1

echo "job ${SLURM_JOB_ID} on ${SLURM_NODELIST}, $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "RUN_CMD: ${RUN_CMD:-<none>}"
echo "------------------------------------------------------------------------"

[ -n "${RUN_CMD:-}" ] || { echo "No RUN_CMD given"; exit 1; }
eval "$RUN_CMD"

echo "------------------------------------------------------------------------"
echo "done $(date)"
