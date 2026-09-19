#!/bin/bash
# Reasoning-batch smoke: 3 conversations to gate the risky arms before the full A/B.
#
#   base         reference
#   reason       terse in-call reasoning field (variant=reason)
#   v1           evidence-first ordering (variant=v1)
#   think        native Gemma-4 thinking at the deployed 1024-token cap
#   think2k      native Gemma-4 thinking at a 2048-token cap (cap diagnosis)
#
# Gate: parse failures, truncated-JSON recoveries, quote-found rate, latency.
# Development-only; no serving, no tunnel, no competition calls. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_reason_smoke
#SBATCH --time=0-00:45:00
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

echo ">>> arm: base"
MEDAPP_LLM_PROMPT=base python llm_probe.py --rungs L1 --limit 3 --tag rs_base

echo ">>> arm: reason"
MEDAPP_LLM_PROMPT=reason python llm_probe.py --rungs L1 --limit 3 --tag rs_reason

echo ">>> arm: v1"
MEDAPP_LLM_PROMPT=v1 python llm_probe.py --rungs L1 --limit 3 --tag rs_v1

echo ">>> arm: think (1024 cap)"
MEDAPP_LLM_PROMPT=base MEDAPP_LLM_ENABLE_THINKING=1 \
    python llm_probe.py --rungs L1 --limit 3 --tag rs_think

echo ">>> arm: think2k (2048 cap)"
MEDAPP_LLM_PROMPT=base MEDAPP_LLM_ENABLE_THINKING=1 MEDAPP_LLM_MAX_NEW_TOKENS=2048 \
    python llm_probe.py --rungs L1 --limit 3 --tag rs_think2k

echo ">>> truncation-recovery signals (stderr)"
grep -c "Recovered .* complete answers" logs/med_reason_smoke_${SLURM_JOB_ID}.err || true

echo "finished $(date)"
