#!/bin/bash
# Provision the local serving environment (GPU box) for the medical-appointment
# endpoint, and pre-download the model weights.
#
#   bash local/setup_env.sh
#
# Separate from the `medical` dev/test env so the serving stack stays isolated.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source /home/dominic/miniforge3/etc/profile.d/conda.sh
if conda env list | awk '{print $1}' | grep -qx medapp-local; then
    echo "conda env 'medapp-local' exists; updating"
else
    echo "creating conda env 'medapp-local'"
    conda create -n medapp-local python=3.11 -y
fi
conda activate medapp-local

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# cu121 wheels support the GTX 1650 (sm_75) in WSL.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install faster-whisper transformers

echo "pre-downloading weights"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Systran/faster-whisper-large-v3")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
AutoTokenizer.from_pretrained(name)
AutoModelForSequenceClassification.from_pretrained(name)
print("local models ready")
PY

echo "medapp-local setup complete"
