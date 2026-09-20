#!/bin/bash
# gpt-oss-120b no-reasoning smoke using a modified chat template.
#
# The generation prompt is pre-filled with an empty closed analysis channel so
# the model must answer directly in the final channel (community workaround,
# HF discussion openai/gpt-oss-20b#86). 3 conversations; latency + parse gate.
# Development-only; no serving, no tunnel, no competition calls.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_gptoss_nr_smoke
#SBATCH --time=0-01:00:00
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

echo ">>> smoke: no-reasoning template, 3 conversations"
MEDAPP_LLM_REASONING_EFFORT=low "${PYTHON}" llm_probe.py --rungs L1 --limit 3 --tag gptoss_nr

echo "finished $(date)"
