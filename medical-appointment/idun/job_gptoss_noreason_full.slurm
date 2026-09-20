#!/bin/bash
# gpt-oss-120b full 39-conversation L1 run with the no-reasoning chat template,
# compared against the recorded base (results/llm_p2_base_L1_questions.json).
# Development-only; no serving, no tunnel, no competition calls.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_gptoss_nr_full
#SBATCH --time=0-01:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export MEDAPP_ANSWERER=llm
export MEDAPP_LLM_MODEL=openai/gpt-oss-120b
export MEDAPP_LLM_REVISION=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
export MEDAPP_LLM_DTYPE=bfloat16
export MEDAPP_LLM_MAX_NEW_TOKENS=1024
export MEDAPP_LLM_LEGACY_SPECIAL_TOKENS=1
export MEDAPP_LLM_CHAT_TEMPLATE=gptoss_no_reasoning.jinja
export MEDAPP_TRANSCRIPT_CONFIG_HASH=e75a7f6e
export WHISPER_MODEL=large-v3-turbo
export WHISPER_COMPUTE_TYPE=int8
PYTHON="${MEDAPP_PYTHON:-$HOME/gptoss-env28/bin/python}"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ">>> full 39: no-reasoning template"
MEDAPP_LLM_REASONING_EFFORT=low "${PYTHON}" llm_probe.py --rungs L1 --tag gptoss_nr

echo ">>> comparison vs base"
python compare_diarization.py --demo-modes first --min-non-demo 30 \
    --base results/llm_p2_base_L1_questions.json \
    --speaker results/llm_gptoss_nr_L1_questions.json || true

echo "finished $(date)"
