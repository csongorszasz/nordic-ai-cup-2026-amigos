#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=drone-serve
#SBATCH --time=2-00:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu32g|gpu40g|gpu80g"
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

# Holds one GPU for serving (idun/serve_ngrok.sh): the job only sleeps; the server runs in a
# tmux session on the node, so it outlives laptop crashes and dropped ssh. scancel frees it.
echo "holding $(hostname) for serving, job ${SLURM_JOB_ID}"
sleep infinity
