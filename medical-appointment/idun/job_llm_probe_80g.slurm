#!/bin/bash
# LLM ceiling probe on one fp16-capable GPU node.
#
#   bash idun/submit.sh llm --rungs L0 L1 L2 --limit 3     # smoke
#   bash idun/submit.sh llm --rungs L0 L1 L2               # full 39
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_llm_probe_80g
#SBATCH --time=0-04:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "########################################################################"

echo ">>> fast test gate"
python -m pytest -m "not slow" -q

echo ">>> llm_probe"
python llm_probe.py "$@"

echo "########################################################################"
echo "finished $(date)"
