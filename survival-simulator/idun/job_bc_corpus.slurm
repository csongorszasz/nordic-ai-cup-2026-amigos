#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=bc_teacher_corpus
#SBATCH --partition=CPUQ
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYGAME_HIDE_SUPPORT_PROMPT=1
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
python -m src.training.demonstrations --config configs/imitation-gru.json \
    --output training-results/bc-teacher-corpus-v1 \
    --train-worlds 8 --validation-worlds 4 --workers 4
