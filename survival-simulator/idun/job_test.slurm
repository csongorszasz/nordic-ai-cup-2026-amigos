#!/bin/bash
# Unit test suite (CPUQ).
#
#   bash idun/submit.sh test
#
# The BENCHMARK_INTEGRATION-gated tests are skipped by default; set that env in
# the job if you want the slow integration runner.
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sv_test
#SBATCH --time=0-01:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"

export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
python -m unittest discover -s tests -p "test_*.py" -v
echo "finished $(date)"
