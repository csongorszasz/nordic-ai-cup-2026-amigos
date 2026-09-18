#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sv_watch
#SBATCH --time=0-12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -m idun.watch_checkpoints "$@"