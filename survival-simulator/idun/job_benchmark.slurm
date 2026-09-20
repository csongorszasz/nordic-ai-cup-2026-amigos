#!/bin/bash
# Native-score benchmark / comparison (CPUQ). Pass benchmark.py arguments.
#
#   bash idun/submit.sh benchmark run --policy random --suite quick --output benchmark-results/random-quick
#   bash idun/submit.sh compare --reference benchmark-results/random-standard \
#       --candidate benchmark-results/ppo-standard --output benchmark-results/cmp
#
# Note: benchmark output directories are never overwritten; use a new
# --output path per run (include the SLURM job id if you like).
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sv_bench
#SBATCH --time=0-12:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs benchmark-results

module purge
module load Anaconda3/2023.09-0
module load CUDA/12.2.0
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-survival}"

export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}   $(date)"
python benchmark.py "$@"
echo "finished $(date)"
