#!/bin/bash
# Multi-candidate selection diagnostic: 26B proposes up to three supporting
# quotes per yes; three selectors (and the oracle) choose one.
#
#   stage 1: multi_probe.py   (26B, incumbent, turbo e75a7f6e)
#   stage 2: multi_select.py  (oracle | s1 self-choice | s2 26B likelihood |
#                              s3a Qwen2.5-7B likelihood)
#   stage 3: paired comparison vs the recorded base arm
#
# Development-only; no serving, no tunnel, no competition calls. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_multi_probe
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

echo ">>> stage 1: generate candidates"
python multi_probe.py --limit 39

for method in oracle s1 s2 s3a; do
    echo ">>> stage 2: select (${method})"
    python multi_select.py --method "${method}"
    echo ">>> compare ${method} - base"
    python compare_diarization.py \
        --base results/llm_p_base_L1_questions.json \
        --speaker "results/multi_${method}_questions.json" \
        --demo-modes first
done

echo "finished $(date)"
