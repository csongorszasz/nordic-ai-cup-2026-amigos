#!/bin/bash
# Train the post-hoc boundary refiner on a conversation-grouped split.
#
#   bash idun/submit.sh refiner --folds 5 --bf16 --tag e4b_5fold
#   bash idun/submit.sh refiner --loco --bf16 --tag e4b_loco
#   bash idun/submit.sh refiner --train-all --bf16 --tag e4b_final
#
# fp16/bf16-capable node only: P100 (gpu16g) cannot do fp16 or int8.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_train_refiner
#SBATCH --time=0-02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu32g|gpu40g|gpu80g"
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs models results transcripts

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

echo ">>> train refiner"
python train_refiner.py "$@"

echo "########################################################################"
echo "finished $(date)"
