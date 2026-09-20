#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=bc_fixed_continue
#SBATCH --partition=GPUQ
#SBATCH --constraint=gpu80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYGAME_HIDE_SUPPORT_PROMPT=1 PYTHONUNBUFFERED=1
python -m src.training.offline_bc --device cuda --steps 10000 --evaluate-every 250 --frames 16 \
  --checkpoint training-results/bc-authorized-25405107/checkpoint.pt \
  --resume training-results/bc-fixed16-25405419/best.pt \
  --output "training-results/bc-fixed16-continue-${SLURM_JOB_ID}"
python -m src.training.offline_bc --device cuda --steps 10000 --evaluate-every 250 \
  --checkpoint training-results/bc-authorized-25405107/checkpoint.pt \
  --resume training-results/bc-fixed160-25405419/best.pt \
  --output "training-results/bc-fixed160-continue-${SLURM_JOB_ID}"
