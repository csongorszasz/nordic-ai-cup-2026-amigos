#!/bin/bash
# ==============================================================================
# Provision the `survival` conda env on IDUN (run on a LOGIN node).
#
#   bash idun/setup_env.sh
#
# The simulation is CPU-bound (Pygame, headless) and neural training adds a
# small PyTorch model. Pinned deps live in requirements*.txt; torch is installed
# from the CUDA index, then cma/matplotlib for training and benchmark plots.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

ENV_NAME="${CONDA_ENV:-survival}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "conda env '$ENV_NAME' exists; updating"
else
    echo "creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# cu128 wheels bundle the CUDA runtime; the module CUDA version is not used by torch.
python -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
python -m pip install -r requirements-training.txt
# Benchmark plotting (optional; compare --no-plots works without it).
python -m pip install -r requirements-benchmark.txt || true

echo ">>> headless simulation smoke"
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    python -c "from local_playground import local_simulation; local_simulation(verbose=False)"

echo ">>> torch / cuda"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "build", torch.version.cuda)
PY

echo "setup complete (env $ENV_NAME)"
