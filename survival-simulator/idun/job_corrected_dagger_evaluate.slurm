#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=dagger_matched_curve
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYGAME_HIDE_SUPPORT_PROMPT=1 PYTHONUNBUFFERED=1
JOB="${1:?Provide the matched fitting job ID}"
python -m idun.watch_checkpoints --training-job "$JOB" --run "training-results/dagger-control-$JOB"
python -m idun.watch_checkpoints --training-job "$JOB" --run "training-results/dagger-dagger-$JOB"
