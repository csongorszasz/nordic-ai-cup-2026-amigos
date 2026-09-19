#!/bin/bash
# Disjoint-occurrence diagnostic smoke for two prompt arms.
#
#   arm A: v1        evidence-first rule + schema, few-shot demos reordered to
#                    evidence_quote-before-answer
#   arm B: v1_reason v1 plus a short "reason" field: reason -> evidence_quote
#                    -> answer, with demos updated to match
#
# Scored on the 16 conversations that contain the 22 positives the recorded base
# arm failed disjointly (base tIoU == 0). All 10 questions are prompted per
# conversation (serving-faithful); only the disjoint positives are scored, by
# compare_subset.py against the known base results/llm_p2_base_L1_questions.json.
# Oracle-selected subset: mechanism diagnostic, not a deployment estimate.
# Development-only; no serving, no tunnel, no competition calls. One 80 GB GPU.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_disjoint_smoke
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

TIDS="sample_4,sample_6,sample_17,sample_19,sample_20,sample_23,sample_33,sample_37,sample_57,sample_63,sample_64,sample_66,sample_70,sample_77,sample_79,sample_82"

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ">>> arm A: v1 (evidence-first, reordered demos)"
MEDAPP_LLM_PROMPT=v1 python llm_probe.py --rungs L1 --only-tids "${TIDS}" --tag dis_v1

echo ">>> arm B: v1_reason (reason -> evidence_quote -> answer)"
MEDAPP_LLM_PROMPT=v1_reason python llm_probe.py --rungs L1 --only-tids "${TIDS}" --tag dis_v1r

echo ">>> disjoint-subset comparison"
python compare_subset.py \
    --base results/llm_p2_base_L1_questions.json \
    --arm results/llm_dis_v1_L1_questions.json \
    --arm results/llm_dis_v1r_L1_questions.json

echo "finished $(date)"
