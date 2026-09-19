#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=drone_test
#SBATCH --time=0-00:20:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g&(a100|h100)"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 20-minute sanity check: GPU visible, env imports, and one YOLO forward pass.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

source idun/runtime.sh

echo "job ${SLURM_JOB_ID} on ${SLURM_NODELIST}, $(date)"
python idun/preflight.py
python - <<'EOF'
import sys, time
import numpy as np, torch
sys.path.insert(0, 'src')
from dtos import OBJECT_CLASSES
from ultralytics import YOLO

print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")
model = YOLO('yolo11s.yaml')
image = np.zeros((540, 960, 3), np.uint8)
model.predict(image, imgsz=960, device=0, verbose=False)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(20):
    model.predict(image, imgsz=960, device=0, verbose=False)
torch.cuda.synchronize()
print(f"YOLO11s 960px: {(time.perf_counter() - start) / 20 * 1000:.1f} ms/frame; {len(OBJECT_CLASSES)} challenge classes")
print("TEST OK")
EOF
