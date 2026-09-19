#!/bin/bash
# System-prompt rule A/B for the incumbent 26B L1 prompt: base vs audit.
#
# Few-shot demonstrations are held identical across arms; only the system-prompt
# rule changes. Served transcript cache (turbo e75a7f6e) and model pinned.
# Development-only; no serving, no tunnel, no competition calls. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_prompt_ab
#SBATCH --time=0-01:00:00
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

export MEDAPP_TRANSCRIPT_CONFIG_HASH=e75a7f6e
export MEDAPP_LLM_MODEL=google/gemma-4-26b-a4b-it
export MEDAPP_LLM_REVISION=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
export MEDAPP_LLM_MAX_NEW_TOKENS=1024
export MEDAPP_LLM_LEGACY_SPECIAL_TOKENS=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ">>> arm A: base"
MEDAPP_LLM_PROMPT=base python llm_probe.py --rungs L1 --tag p_base

echo ">>> arm B: audit"
MEDAPP_LLM_PROMPT=audit python llm_probe.py --rungs L1 --tag p_audit

echo ">>> paired comparison (audit - base)"
python compare_diarization.py \
    --base results/llm_p_base_L1_questions.json \
    --speaker results/llm_p_audit_L1_questions.json

echo "finished $(date)"
