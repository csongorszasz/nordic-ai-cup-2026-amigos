#!/bin/bash
# Recall diagnostic: proposition entailment vs clause / clause+/-1 / gold text,
# for both base and large NLI.
#
#   sbatch --account=share-ie-idi idun/job_diag.slurm
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_diag
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs transcripts results

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export MEDAPP_SKIP_WARMUP=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ">>> fast test gate"
python -m pytest -m "not slow" -q

BASE="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LARGE="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

echo "########################################################################"
echo ">>> base NLI"
NLI_MODEL="$BASE" MEDAPP_NLI_TAU=0.3 python diagnose_recall.py --tag base

echo "########################################################################"
echo ">>> large NLI"
NLI_MODEL="$LARGE" MEDAPP_NLI_TAU=0.3 python diagnose_recall.py --tag large

echo "########################################################################"
echo "finished $(date)"
