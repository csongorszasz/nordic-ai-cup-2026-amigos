#!/bin/bash
# Neural training: imitation (behavioral cloning + DAgger) or PPO.
#
#   bash idun/submit.sh train configs/ppo-gru.json ppo
#   bash idun/submit.sh train configs/imitation-gru.json imitation
#
# Arguments:
#   $1  experiment config JSON (e.g. configs/ppo-gru.json)
#   $2  run name (output dir training-results/<name>-<jobid>)
#   ... extra train.py args (e.g. --resume <checkpoint>)
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sv_train
#SBATCH --time=0-12:00:00
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
mkdir -p logs training-results

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"

export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1

CONFIG="${1:-configs/ppo-gru.json}"
NAME="${2:-train}"
shift 2 || true
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "config=${CONFIG} name=${NAME}; worker and memory settings come from the reviewed run plan"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python train.py \
    --config "${CONFIG}" \
    --output "training-results/${NAME}-${SLURM_JOB_ID}" \
    "$@"

echo "finished $(date)"
