#!/bin/bash
# Provision the `nordic` conda environment on IDUN and pre-download ASR weights.
#
# Run on an IDUN LOGIN node (compute nodes may lack internet):
#   bash idun/setup_env.sh
#
# Weights land in the shared HF cache (~/.cache/huggingface), which compute
# nodes can read; jobs then run with HF_HUB_OFFLINE=1.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx nordic; then
    echo "conda env 'nordic' exists; updating"
else
    echo "creating conda env 'nordic'"
    conda create -n nordic python=3.11 -y
fi
conda activate nordic

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-model.txt
# torch brings the CUDA runtime libs (cuBLAS/cuDNN) that ctranslate2 needs.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install faster-whisper

echo "pre-downloading faster-whisper weights"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-large-v3")
PY

echo "pre-downloading NLI model"
python - <<'PY'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
AutoTokenizer.from_pretrained(name)
AutoModelForSequenceClassification.from_pretrained(name)
print("downloaded", name)
PY

echo "setup complete"
