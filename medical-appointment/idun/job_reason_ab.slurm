#!/bin/bash
# Reasoning-batch full A/B for the incumbent 26B L1 prompt.
#
#   arm A: base     reference
#   arm B: reason   terse in-call reasoning field
#   arm C: v1       evidence-first ordering
#   arm D: think    native Gemma-4 thinking (deployed 1024-token cap)
#
# Demonstrations, model revision, ASR cache (turbo e75a7f6e), decoding and output
# schema are held identical across arms; only the system rule / thinking flag
# changes. Comparisons pin the demonstration set to the base "first" selection so
# the demonstration-disjoint cohort stays at ~35 conversations.
# Development-only; no serving, no tunnel, no competition calls. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_reason_ab
#SBATCH --time=0-02:00:00
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
MEDAPP_LLM_PROMPT=base python llm_probe.py --rungs L1 --tag r_base

echo ">>> arm B: reason"
MEDAPP_LLM_PROMPT=reason python llm_probe.py --rungs L1 --tag r_reason

echo ">>> arm C: v1"
MEDAPP_LLM_PROMPT=v1 python llm_probe.py --rungs L1 --tag r_v1

echo ">>> arm D: think"
MEDAPP_LLM_PROMPT=base MEDAPP_LLM_ENABLE_THINKING=1 \
    python llm_probe.py --rungs L1 --tag r_think

echo ">>> paired comparison: reason - base"
python compare_diarization.py --demo-modes first --min-non-demo 30 \
    --base results/llm_r_base_L1_questions.json \
    --speaker results/llm_r_reason_L1_questions.json

echo ">>> paired comparison: v1 - base"
python compare_diarization.py --demo-modes first --min-non-demo 30 \
    --base results/llm_r_base_L1_questions.json \
    --speaker results/llm_r_v1_L1_questions.json

echo ">>> paired comparison: think - base"
python compare_diarization.py --demo-modes first --min-non-demo 30 \
    --base results/llm_r_base_L1_questions.json \
    --speaker results/llm_r_think_L1_questions.json

echo "finished $(date)"
