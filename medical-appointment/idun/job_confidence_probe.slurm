#!/bin/bash
# Citation-confidence diagnostic for the incumbent 26B L1 prompt.
#
# Runs the serving prompt once per conversation with output_scores, attributes
# the completion log-probability to each positive question, and reports whether
# confidence separates localization hits from misses. Development-only; no
# serving, no tunnel, no competition calls. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_conf_probe
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

# Served configuration: turbo int8 transcripts, incumbent 26B, base prompt.
export MEDAPP_TRANSCRIPT_CONFIG_HASH=e75a7f6e
export MEDAPP_LLM_MODEL=google/gemma-4-26b-a4b-it
export MEDAPP_LLM_REVISION=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
export MEDAPP_LLM_MAX_NEW_TOKENS=1024
export MEDAPP_LLM_LEGACY_SPECIAL_TOKENS=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python confidence_probe.py --tag L1_turbo

echo "finished $(date)"
