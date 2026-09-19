#!/bin/bash
set -euo pipefail
module purge
module load Anaconda3/2023.09-0
: "${DRONE_ENV_PATH:?Set DRONE_ENV_PATH to the experiment environment}"
source "${DRONE_ENV_PATH}/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}/runs/ultralytics-config"
mkdir -p "$YOLO_CONFIG_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
