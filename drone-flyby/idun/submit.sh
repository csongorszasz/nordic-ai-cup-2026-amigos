#!/bin/bash
# ==============================================================================
# Run drone-flyby training on NTNU IDUN from a laptop.
# Adapted from Juan's thesis-prep harness (src/idun/).
#
# Usage (from drone-flyby/):
#   bash idun/submit.sh setup                 # once: build the conda env on IDUN
#   bash idun/submit.sh test                  # 20-min job: GPU, torch, ultralytics sanity
#   bash idun/submit.sh run <command...>      # GPU job running any command in drone-flyby/
#   bash idun/submit.sh train-synth [frames]  # synthetic dataset + YOLO training, one job
#   bash idun/submit.sh queue                 # your jobs
#   bash idun/submit.sh logs [job_id]         # tail the latest (or one) job log
#   bash idun/submit.sh fetch                 # pull runs/ (weights, metrics) back here
#   bash idun/submit.sh sync                  # upload code without submitting
#
# Examples:
#   bash idun/submit.sh run python training/train_yolo.py --epochs 60 --batch 32
#   bash idun/submit.sh train-synth 400
#
# Overrides:
#   REMOTE=idun                        SSH alias from ~/.ssh/config
#   REMOTE_DIR=~/nordic-cup/drone-flyby  kept apart from anything else in the home dir
#   SLURM_ACCOUNT=share-ie-idi         allocation to bill
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRONE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-idun}"
REMOTE_DIR="${REMOTE_DIR:-~/nordic-cup/drone-flyby}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-share-ie-idi}"

ACTION="${1:-help}"

check_ssh() {
    echo -n "Checking SSH connection to '${REMOTE}'... "
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" "echo ok" >/dev/null 2>&1; then
        echo "FAILED"
        echo "Cannot reach '${REMOTE}'. Are you on eduroam or the NTNU VPN, and is"
        echo "'Host ${REMOTE}' in ~/.ssh/config? See idun/README.md."
        exit 1
    fi
    echo "OK"
}

# Code, the supplied scene, sprite cut-outs and backgrounds go up; generated
# datasets, runs, recordings and the local venv stay here. Datasets are rebuilt
# on IDUN, which is faster than uploading them. Two exceptions: the 3D-model sprite
# bank goes up (rendering it needs OpenGL), and the raw city mesh stays here (GBs;
# only its rendered frames in backgrounds/helsinki3d_frames/ are needed).
sync_code() {
    ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}"
    echo "Syncing ${DRONE_DIR} -> ${REMOTE}:${REMOTE_DIR}"
    rsync -az --delete --info=stats1 \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        --exclude='.downloads' \
        --include='/datasets/' \
        --include='/datasets/model_sprites/***' \
        --exclude='/datasets/*' \
        --exclude='/backgrounds/helsinki3d/' \
        --exclude='runs' \
        --exclude='recordings' \
        --exclude='logs' \
        --exclude='annotated' \
        --exclude='*.pt' \
        "${DRONE_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
}

submit() {
    local job_file="$1" cmd="$2"
    local out
    out=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && RUN_CMD=$(printf '%q' "$cmd") sbatch --export=ALL --account=${SLURM_ACCOUNT} idun/${job_file}")
    echo "$out"
    local job_id
    job_id=$(awk '{print $NF}' <<<"$out")
    echo ""
    echo "Follow it with:  bash idun/submit.sh logs ${job_id}"
}

case "$ACTION" in
    sync)
        check_ssh
        sync_code
        ;;

    setup)
        check_ssh
        sync_code
        ssh -t "$REMOTE" "cd ${REMOTE_DIR} && bash idun/setup_env.sh"
        ;;

    test)
        check_ssh
        sync_code
        submit job_test.slurm ""
        ;;

    run)
        shift
        [ $# -gt 0 ] || { echo "run needs a command, e.g. python training/train_yolo.py"; exit 1; }
        check_ssh
        sync_code
        submit job.slurm "$*"
        ;;

    train-synth)
        FRAMES="${2:-300}"
        check_ssh
        sync_code
        # Synthetic only; the real Helsinki scene (all 25 frames) is the validation set, so the
        # best checkpoint is picked on real imagery. Copenhagen stays out of it entirely.
        submit job.slurm "python training/make_dataset.py --all-val --out datasets/helsinki_real && python training/synth_dataset.py --frames ${FRAMES} --val-dir datasets/helsinki_real/images/val && python training/train_yolo.py --data datasets/synth_yolo/data.yaml --epochs 60 --batch 32 --name synth_${FRAMES}"
        ;;

    queue)
        check_ssh
        ssh "$REMOTE" "squeue -u \$(whoami) --format='%.10i %.9P %.22j %.2t %.10M %.6D %R'"
        ;;

    logs)
        check_ssh
        TARGET="${2:-}"
        if [ -n "$TARGET" ]; then
            ssh -t "$REMOTE" "tail -n 50 -f ${REMOTE_DIR}/logs/*${TARGET}*.out"
        else
            LATEST=$(ssh "$REMOTE" "ls -t ${REMOTE_DIR}/logs/*.out 2>/dev/null | head -n 1" || true)
            [ -n "$LATEST" ] || { echo "No logs yet in ${REMOTE}:${REMOTE_DIR}/logs/"; exit 0; }
            echo "Tailing ${LATEST}"
            ssh -t "$REMOTE" "tail -n 50 -f ${LATEST}"
        fi
        ;;

    fetch)
        check_ssh
        mkdir -p "${DRONE_DIR}/runs"
        rsync -az --info=stats1 "${REMOTE}:${REMOTE_DIR}/runs/" "${DRONE_DIR}/runs/"
        echo "Weights are in runs/<name>/weights/best.pt. Serve one with DRONE_MODEL=<path>."
        ;;

    help|--help|-h)
        sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        ;;

    *)
        echo "Unknown action '$ACTION'. Run: bash idun/submit.sh help"
        exit 1
        ;;
esac
