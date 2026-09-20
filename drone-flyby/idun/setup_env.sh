#!/bin/bash
# Build or update the `drone` conda env on IDUN. Run through `bash idun/submit.sh setup`.
# Same module stack as the thesis harness, which is known to work on IDUN.
set -euo pipefail

ENV_NAME="drone"
DRONE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== ${ENV_NAME} env on $(hostname), $(date) ==="
module purge
module load Anaconda3/2023.09-0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh

if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Env '${ENV_NAME}' exists, updating packages"
else
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# torch 2.5.1 + CUDA 12.1 wheels: the same versions we train with locally, and they
# bring their own CUDA runtime, so no CUDA module is needed inside the job.
pip install --upgrade pip
pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics -r "${DRONE_DIR}/requirements.txt"

# Pretrained weights are gitignored and kept out of the sync (excluded files also survive
# rsync --delete), so fetch them here on the login node, which has internet access.
(cd "$DRONE_DIR" && python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')")

python - <<'EOF'
import torch, ultralytics, cv2
print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}), ultralytics {ultralytics.__version__}, cv2 {cv2.__version__}")
print("GPU visible here:", torch.cuda.is_available(), "(False is normal on a login node)")
EOF
echo "=== done: jobs activate it with 'conda activate ${ENV_NAME}' ==="
