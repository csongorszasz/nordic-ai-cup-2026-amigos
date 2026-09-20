#!/bin/bash
# Throughput profile: measure transitions/s and peak frame tokens (CPUQ).
#
#   bash idun/submit.sh profile
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sv_profile
#SBATCH --time=0-01:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
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

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
python train.py \
    --config configs/profile.json \
    --set resources.workers="${WORKERS}" \
    --output "training-results/profile-${SLURM_JOB_ID}" \
    "$@"
echo "finished $(date)"
