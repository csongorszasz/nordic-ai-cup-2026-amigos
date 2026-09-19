#!/bin/bash
# Create a task-owned environment; never update a shared conda environment.
set -euo pipefail

DRONE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DRONE_ENV_PATH:?Set DRONE_ENV_PATH to a new task-owned environment}"

echo "=== ${DRONE_ENV_PATH} on $(hostname), $(date) ==="
module purge
module load Anaconda3/2023.09-0
if [ -f "${DRONE_ENV_PATH}/.ready" ]; then
    echo "Environment already initialized; use a new path to change dependencies."
    exit 0
fi
python -m venv "${DRONE_ENV_PATH}"
source "${DRONE_ENV_PATH}/bin/activate"
export YOLO_CONFIG_DIR="$DRONE_DIR/runs/ultralytics-config"
mkdir -p "$YOLO_CONFIG_DIR"

# torch 2.5.1 + CUDA 12.1 wheels: the same versions we train with locally, and they
# bring their own CUDA runtime, so no CUDA module is needed inside the job.
python -m pip install --quiet "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install --quiet -r "${DRONE_DIR}/requirements.txt" "numpy<2" pytest httpx
python -m pip check
python -m pip freeze > "${DRONE_ENV_PATH}/requirements.lock"
python "${DRONE_DIR}/idun/preflight.py" --configure-only

# Pretrained weights are gitignored and kept out of the sync (excluded files also survive
# rsync --delete), so fetch them here on the login node, which has internet access.
(cd "$DRONE_DIR" && python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')")

python - <<'EOF'
import torch, ultralytics, cv2
print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}), ultralytics {ultralytics.__version__}, cv2 {cv2.__version__}")
print("GPU visible here:", torch.cuda.is_available(), "(False is normal on a login node)")
EOF
touch "${DRONE_ENV_PATH}/.ready"
echo "=== environment ready ==="
