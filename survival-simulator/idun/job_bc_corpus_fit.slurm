#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=bc_offline_fit
#SBATCH --partition=GPUQ
#SBATCH --constraint=gpu80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYGAME_HIDE_SUPPORT_PROMPT=1 PYTHONUNBUFFERED=1
CHECKPOINT="${1:?Provide the qualified fixed-data checkpoint}"
shift
python -m src.training.bc_corpus_fit --corpus training-results/bc-teacher-corpus-v1 \
  --checkpoint "${CHECKPOINT}" --output "training-results/bc-corpus-fit-${SLURM_JOB_ID}" \
  --steps 1000 --sequence-length 64 --burn-in 256 --batch-sequences 4 \
  --validate-every 250 --device cuda --max-vram-mb 6144 --max-seconds 1000 "$@"
