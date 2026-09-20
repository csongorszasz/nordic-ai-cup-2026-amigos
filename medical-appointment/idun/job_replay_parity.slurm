#!/bin/bash
# Replay-parity diagnostic: does the frozen endpoint apply the +0.2 calibration?
#
# Replays two captured validation conversations through the local pipeline with
# and without MEDAPP_SPAN_CALIBRATION and compares the spans to the endpoint's
# stored captures. Development-only; captures disabled; no serving/tunnel.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_replay_parity
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
export MEDAPP_ANSWERER=llm
export MEDAPP_LLM_MODEL=google/gemma-4-26b-a4b-it
export MEDAPP_LLM_REVISION=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
export MEDAPP_LLM_MAX_NEW_TOKENS=1024
export MEDAPP_LLM_LEGACY_SPECIAL_TOKENS=1
export WHISPER_MODEL=large-v3-turbo
export WHISPER_COMPUTE_TYPE=int8
export MEDAPP_CAPTURE=0
unset MEDAPP_INFERENCE_WORKER

CONVS="sample_3,sample_80"
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ">>> raw (no calibration)"
unset MEDAPP_SPAN_CALIBRATION
python replay_parity.py run --tag raw --convs "${CONVS}"

echo ">>> calibrated (+0.2)"
MEDAPP_SPAN_CALIBRATION=calibration/span_offset_base.json \
    python replay_parity.py run --tag cal --convs "${CONVS}"

echo ">>> compare endpoint vs raw vs calibrated"
python replay_parity.py compare --tags raw cal

echo "finished $(date)"
