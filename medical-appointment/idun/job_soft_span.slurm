#!/bin/bash
# Soft-target span tagger: ModernBERT token classifier trained on a soft
# per-word evidence label from gold + agent annotations, evaluated OOF.
#
# Development-only; decisions stay fixed. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_soft_span
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --mem=64G
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
export MEDAPP_TRANSCRIPT_CONFIG_HASH=e75a7f6e

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ">>> train soft-target tagger (OOF)"
python soft_span_model.py --folds 5 --epochs 4

echo ">>> paired comparison: soft-span OOF - base"
python compare_diarization.py \
    --base results/llm_p_base_L1_questions.json \
    --speaker results/soft_span_questions.json \
    --demo-modes first

echo "finished $(date)"
