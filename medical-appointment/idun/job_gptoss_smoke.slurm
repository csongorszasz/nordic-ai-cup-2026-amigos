#!/bin/bash
# gpt-oss-120b load/generate smoke on one 80 GB GPU. Development-only.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_gptoss_smoke
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
# Run-local overlay (system-site-packages) that adds accelerate for MXFP4; the
# shared nordic env is not modified.
PYTHON="${MEDAPP_PYTHON:-$HOME/gptoss-env/bin/python}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PYTHON}" idun/gptoss_load_smoke.py
echo "finished $(date)"
