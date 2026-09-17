#!/bin/bash
# End-to-end scoring job: start the API on this compute node, then replay the
# supplied conversations through it with the official evaluator.
#
#   sbatch --account=share-ie-idi idun/job_eval.slurm
#
# One GPU is requested; the current baseline does not use it, but the ASR stack
# will. Everything runs on the node, so no public networking is involved.

#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_eval
#SBATCH --time=0-00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs transcripts models

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

echo "========================================================================"
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "========================================================================"

python - <<'PY'
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)
try:
    import ctranslate2
    print("ctranslate2", ctranslate2.__version__,
          "cuda devices", ctranslate2.get_cuda_device_count())
except Exception as e:
    print("ctranslate2 import failed:", e)
PY

# Start the endpoint on this node and score it against the supplied data.
python api.py > "logs/api_${SLURM_JOB_ID}.log" 2>&1 &
API_PID=$!
trap 'kill ${API_PID} 2>/dev/null || true' EXIT

python local_evaluator.py --verbose

echo "========================================================================"
echo "finished $(date)"
