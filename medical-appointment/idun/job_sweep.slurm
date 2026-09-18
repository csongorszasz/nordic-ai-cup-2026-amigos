#!/bin/bash
# Tuning sweep: several dev_eval configurations in one GPU allocation.
# Transcripts are cached, so each run is verifier-only.
#
#   sbatch --account=share-ie-idi idun/job_sweep.slurm
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_sweep
#SBATCH --time=0-01:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu32g|gpu40g|gpu80g"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs transcripts results

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export MEDAPP_SKIP_WARMUP=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run() {
    tag="$1"; shift
    echo ""
    echo "########################################################################"
    echo "### ${tag}: $*"
    echo "########################################################################"
    env "$@" python dev_eval.py --answer nli --diagnostics --json "results/${tag}.json" --tag "${tag}"
}

run_oof() {
    tag="$1"; shift
    echo ""
    echo "########################################################################"
    echo "### ${tag} (OOF, tau=0)"
    echo "########################################################################"
    env "$@" python dev_eval.py --answer nli --json "results/${tag}.json" --tag "${tag}" --oof "results/oof_${tag}.json"
}

BASE="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LARGE="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# New length-diverse candidate cap (C), served caps, merged decision.
run T028 NLI_MODEL="$BASE"  MEDAPP_NLI_TAU=0.3 MEDAPP_DECISION_NEIGHBOURS=1 MEDAPP_TOP_CLAUSES=1 MEDAPP_SELECT=greedy_trim MEDAPP_TRIM_MARGIN=0.1 MEDAPP_MAX_CANDIDATES=120 MEDAPP_MAX_RANGE_WORDS=10
run T029 NLI_MODEL="$LARGE" MEDAPP_NLI_TAU=0.3 MEDAPP_DECISION_NEIGHBOURS=1 MEDAPP_TOP_CLAUSES=1 MEDAPP_SELECT=greedy_trim MEDAPP_TRIM_MARGIN=0.1 MEDAPP_MAX_CANDIDATES=120 MEDAPP_MAX_RANGE_WORDS=10
# OOF records for threshold calibration (B); tau=0 forces every question to localize.
run_oof T030 NLI_MODEL="$LARGE" MEDAPP_NLI_TAU=0.0 MEDAPP_DECISION_NEIGHBOURS=1 MEDAPP_TOP_CLAUSES=1 MEDAPP_SELECT=greedy_trim MEDAPP_TRIM_MARGIN=0.1 MEDAPP_MAX_CANDIDATES=120 MEDAPP_MAX_RANGE_WORDS=10

echo "########################################################################"
echo "finished $(date)"
