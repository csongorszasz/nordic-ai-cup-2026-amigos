#!/bin/bash
# Provision the LLM ceiling-probe dependencies on IDUN (run on a LOGIN node).
#
#   bash idun/setup_llm.sh
#
# Uses the existing `nordic` conda env (no vLLM: the probe is sequential and
# transformers is already installed). Pre-downloads the model so GPU jobs can
# run with HF_HUB_OFFLINE=1.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

python -m pip install sentencepiece

echo "pre-downloading Qwen/Qwen2.5-7B-Instruct"
python - <<'PY'
from huggingface_hub import snapshot_download
name = "Qwen/Qwen2.5-7B-Instruct"
snapshot_download(name)
print("downloaded", name)
PY

echo "setup_llm complete"
