#!/bin/bash
# Dev-eval job: run the fast test gate, then score the pipeline in-process on a
# GPU node. Transcripts are cached, so repeat runs are verifier-only.
#
#   sbatch --account=share-ie-idi idun/job_dev_eval.slurm --diagnostics
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_dev_eval
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
mkdir -p logs transcripts

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
echo "########################################################################"

echo ">>> fast test gate"
python -m pytest -m "not slow" -q

echo ">>> dev_eval"
python dev_eval.py "$@"

echo "########################################################################"
echo "finished $(date)"
