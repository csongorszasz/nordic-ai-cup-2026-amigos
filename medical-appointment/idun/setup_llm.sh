#!/bin/bash
# Provision the LLM serving/probe dependencies on IDUN (run on a LOGIN node).
#
#   bash idun/setup_llm.sh            # Gemma 4 E4B (the validated model)
#   bash idun/setup_llm.sh --with-26b # also Gemma 4 26B-A4B (~52 GB)
#
# Uses the existing `nordic` conda env. Pre-downloads the models so GPU jobs can
# run with HF_HUB_OFFLINE=1, and records what was downloaded so the evaluation
# runs exactly the validated weights and packages:
#
#   models/llm_revisions.txt     "<model id> <commit hash>" per model; the serve
#                                job pins MEDAPP_LLM_REVISION from it
#   idun/serving-env.lock.txt    pip freeze of the env; commit it

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic

python -m pip install sentencepiece

MODELS=("google/gemma-4-e4b-it")
[ "${1:-}" = "--with-26b" ] && MODELS+=("google/gemma-4-26b-a4b-it")

mkdir -p models
for name in "${MODELS[@]}"; do
    echo "pre-downloading ${name}"
    python - "$name" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

name = sys.argv[1]
path = snapshot_download(name)
revision = Path(path).name  # .../snapshots/<commit hash> = exactly what is cached
lines = [
    line for line in Path("models/llm_revisions.txt").read_text().splitlines()
    if line.split() and line.split()[0] != name
] if Path("models/llm_revisions.txt").exists() else []
lines.append(f"{name} {revision}")
Path("models/llm_revisions.txt").write_text("\n".join(lines) + "\n")
print("downloaded", name, "revision", revision)
PY
done

python -m pip freeze > idun/serving-env.lock.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
echo "revisions: $(tr '\n' ';' < models/llm_revisions.txt)"
echo "setup_llm complete — commit idun/serving-env.lock.txt"
