#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=dagger_collect
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYGAME_HIDE_SUPPORT_PROMPT=1 PYTHONUNBUFFERED=1
python -m src.training.dagger_collection anchor \
  --checkpoint training-results/bc-corpus-fit-25406057/checkpoint.pt \
  --expert training-results/bc-teacher-corpus-v1 \
  --output training-results/dagger-comparison-anchor-v1 --peer-context
python -m src.training.dagger_collection collect \
  --checkpoint training-results/dagger-comparison-anchor-v1/checkpoint.pt \
  --expert training-results/bc-teacher-corpus-v1 \
  --output training-results/dagger-recovery-v1 \
  --episodes 2 --teacher-probability 0.5 --workers 2
python -m src.training.dagger_collection aggregate \
  --expert training-results/bc-teacher-corpus-v1 --recovery training-results/dagger-recovery-v1 \
  --output training-results/dagger-aggregate-v1
