#!/bin/bash
# ASR smoke test on one GPU node: transcribe a few conversations and print the
# transcript next to the gold questions.
#
#   sbatch --account=share-ie-idi idun/job_smoke.slurm
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_asr_smoke
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs transcripts

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "########################################################################"

python idun/asr_smoke.py sample_4 sample_17 sample_10

echo "########################################################################"
echo "finished $(date)"
