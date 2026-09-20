#!/bin/bash
# CMA controller search (CPU-bound, multiprocess).
#
#   bash idun/submit.sh search --set search.candidates=16 --set search.worlds=4
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sv_search
#SBATCH --time=0-12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs training-results

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"

export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1

# Leave one core for the collector; the code pins child thread limits itself.
WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
echo "workers=${WORKERS}"

python train.py \
    --config configs/search.json \
    --set resources.workers="${WORKERS}" \
    --output "training-results/search-${SLURM_JOB_ID}" \
    "$@"

echo "finished $(date)"
