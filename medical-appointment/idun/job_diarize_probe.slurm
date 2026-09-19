#!/bin/bash
# Offline speaker-diarization A/B probe for the incumbent 26B LLM.
#
#   1. diarize all 39 conversations (local WavLM x-vector, no network)
#   2. L1 probe with MEDAPP_LLM_SPEAKERS unset  (baseline)
#   3. L1 probe with MEDAPP_LLM_SPEAKERS=1       (speaker-tagged transcript)
#   4. paired comparison on demonstration-disjoint conversations
#
# Development-only: no serving, no tunnel, no competition calls. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_dia_probe
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

# Isolated A/B configuration: served transcript cache, raw predictions only.
export MEDAPP_TRANSCRIPT_CONFIG_HASH=e75a7f6e
unset MEDAPP_SPAN_CALIBRATION
export MEDAPP_LLM_MODEL=google/gemma-4-26b-a4b-it
export MEDAPP_LLM_REVISION=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
export MEDAPP_LLM_MAX_NEW_TOKENS=1024
export MEDAPP_LLM_LEGACY_SPECIAL_TOKENS=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "effective config: transcript=${MEDAPP_TRANSCRIPT_CONFIG_HASH} calibration=${MEDAPP_SPAN_CALIBRATION:-none} model=${MEDAPP_LLM_MODEL} revision=${MEDAPP_LLM_REVISION} tokens=${MEDAPP_LLM_MAX_NEW_TOKENS} legacy=${MEDAPP_LLM_LEGACY_SPECIAL_TOKENS}"

echo ">>> diarize 39 conversations"
python diarize_conversations.py --model models/wavlm-base-plus-sv --limit 39

echo ">>> probe A: baseline (speaker tags explicitly off)"
MEDAPP_LLM_SPEAKERS=0 python llm_probe.py --rungs L1 --tag dia_base

echo ">>> probe B: speaker-tagged transcript (strict sidecars)"
MEDAPP_LLM_SPEAKERS=1 python llm_probe.py --rungs L1 --tag dia_speaker

echo ">>> paired comparison"
python compare_diarization.py \
    --base results/llm_dia_base_L1_questions.json \
    --speaker results/llm_dia_speaker_L1_questions.json \
    --demo-modes first --input-state raw

echo "finished $(date)"
