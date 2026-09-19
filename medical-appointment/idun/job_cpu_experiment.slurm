#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=med_cpu
#SBATCH --time=0-00:30:00
#SBATCH --partition=CPUQ
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module purge
module load Anaconda3/2023.09-0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nordic
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEDAPP_SKIP_WARMUP=1
export MEDAPP_CAPTURE=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -m "not slow" -q
python - <<'PY'
import json
import os
from pathlib import Path
request = json.loads(Path("run_request.json").read_text())
os.environ.update(request["environment"])
python = request["environment"].get("MEDAPP_PYTHON", os.sys.executable)
if not Path(python).is_file():
    raise FileNotFoundError(f"Configured experiment interpreter is missing: {python}")
os.execv(python, [python, request["script"], *request["arguments"]])
PY
