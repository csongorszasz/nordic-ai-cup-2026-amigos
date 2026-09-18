#!/bin/bash
# ==============================================================================
# Run drone-flyby training on NTNU IDUN from a laptop.
# Adapted from Juan's thesis-prep harness (src/idun/).
#
# Usage (from drone-flyby/):
#   bash idun/submit.sh setup                 # once: build the conda env on IDUN
#   bash idun/submit.sh test                  # 20-min job: GPU, torch, ultralytics sanity
#   bash idun/submit.sh run <command...>      # GPU job running any command in drone-flyby/
#   bash idun/submit.sh train-exact [epochs] [batch]  # exact L0/L1/L2 views + YOLO11s@960
#   bash idun/submit.sh queue                 # your jobs
#   bash idun/submit.sh logs [job_id]         # tail the latest (or one) job log
#   bash idun/submit.sh fetch                 # pull runs/ (weights, metrics) back here
#   bash idun/submit.sh sync                  # upload code without submitting
#
# Examples:
#   bash idun/submit.sh run python src/offline/train_yolo.py --data-yaml <yaml> --epochs 60 --batch 32
#   bash idun/submit.sh train-exact 150 32
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

# Code and the supplied scene go up; generated datasets, checkpoints, runs and
# the local venv stay here. The exact-view dataset is rebuilt on IDUN, which is
# faster than uploading it.
sync_code() {
    ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}"
    echo "Syncing ${DRONE_DIR} -> ${REMOTE}:${REMOTE_DIR}"
    rsync -az --delete --info=stats1 \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        --exclude='.downloads' \
        --exclude='datasets' \
        --exclude='training_artifacts' \
        --exclude='runs' \
        --exclude='recordings' \
        --exclude='logs' \
        --exclude='annotated' \
        --include='weights/' \
        --include='weights/***' \
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
        [ $# -gt 0 ] || { echo "run needs a command, e.g. python src/offline/train_yolo.py --data-yaml <yaml>"; exit 1; }
        check_ssh
        sync_code
        submit job.slurm "$*"
        ;;

    # Reproduce the laptop training run on IDUN: build evaluator-exact L0/L1/L2
    # views (873 train / 147 val at 960x540), then train YOLO11s at 960. The
    # dataset is generated on the cluster so only data/helsinki/ is uploaded.
    train-exact)
        EPOCHS="${2:-150}"
        BATCH="${3:-32}"
        BASE_WEIGHTS="${BASE_WEIGHTS:-weights/yolo11s_drone_flyby.pt}"
        check_ssh
        sync_code
        submit job.slurm "python src/offline/build_exact_view_dataset.py --output-dir training_artifacts/exact_views && python src/offline/train_yolo.py --data-yaml training_artifacts/exact_views/drone_flyby_exact/drone_flyby_exact.yaml --weights ${BASE_WEIGHTS} --imgsz 960 --epochs ${EPOCHS} --batch ${BATCH} --device 0 --optimizer auto --patience 40 --cache --project \$PWD/runs/exact_runs --name yolo11s_exact${EPOCHS}"
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
        echo "Weights are in runs/exact_runs/<name>/weights/best.pt."
        echo "Serve one with DRONE_FLYBY_YOLO_WEIGHTS_PATH=<path>."
        ;;

    help|--help|-h)
        sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        ;;

    *)
        echo "Unknown action '$ACTION'. Run: bash idun/submit.sh help"
        exit 1
        ;;
esac
