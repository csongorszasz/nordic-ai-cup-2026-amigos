#!/bin/bash
# Short network-feasibility test on a GPU node: compute egress + cloudflared.
#
#   sbatch --account=share-ie-idi idun/job_net_test.slurm
#
# fp16-capable node (same class we would serve on).
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_net_e2e
#SBATCH --time=0-00:10:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu32g|gpu40g|gpu80g"
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
bash idun/net_e2e_check.sh
echo "finished $(date)"
