#!/bin/bash
# ==============================================================================
# IDUN deployment & SLURM submission harness for survival-simulator
# ==============================================================================
#
# Usage (from the survival-simulator project root):
#   bash idun/submit.sh setup                       # create/update the conda env
#   bash idun/submit.sh sync                        # rsync project to IDUN
#   bash idun/submit.sh search [train.py args]      # CMA controller search (CPUQ)
#   bash idun/submit.sh profile [train.py args]     # throughput profile (CPUQ)
#   bash idun/submit.sh train <config> <name> [args]  # imitation/PPO (GPUQ)
#   bash idun/submit.sh benchmark [benchmark.py args] # suite run (CPUQ)
#   bash idun/submit.sh compare [benchmark.py args]   # compare saved runs
#   bash idun/submit.sh test                        # unittest suite (CPUQ)
#   bash idun/submit.sh serve [policy-config.json]  # agent_server + cloudflared
#   bash idun/submit.sh pull                        # copy training/benchmark results back
#   bash idun/submit.sh queue                       # show your SLURM jobs
#   bash idun/submit.sh logs [id]                   # tail a job log
#
# Environment overrides:
#   REMOTE="idun"                    SSH alias from ~/.ssh/config
#   REMOTE_DIR="~/nordic-survival"   target dir on IDUN
#   SLURM_ACCOUNT="share-ie-idi"     SLURM allocation
#   CONDA_ENV="survival"             conda env name on IDUN
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-idun}"
REMOTE_DIR="${REMOTE_DIR:-~/nordic-survival}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-share-ie-idi}"
CONDA_ENV="${CONDA_ENV:-survival}"

ACTION="${1:-help}"

check_ssh() {
    echo -n "Checking SSH connection to '${REMOTE}'... "
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" "echo ok" >/dev/null 2>&1; then
        echo "FAILED"
        echo "  Cannot connect. IDUN requires the NTNU VPN/eduroam and a registered SSH key."
        exit 1
    fi
    echo "OK"
}

sync_code() {
    echo "Syncing ${PROJECT_ROOT} -> ${REMOTE}:${REMOTE_DIR}"
    ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}"
    rsync -az --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='training-results' \
        --exclude='benchmark-results' \
        "${PROJECT_ROOT}/" "${REMOTE}:${REMOTE_DIR}/"
    echo "Sync complete."
}

sync_training_release() {
    # A reviewed run gets its own immutable source directory, not the live shared tree.
    local plan="$1"
    local plan_id
    plan_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_id"])' "$plan")
    [[ "$plan_id" =~ ^[a-f0-9]{64}$ ]] || { echo "Invalid run-plan ID"; exit 1; }
    local base="$REMOTE_DIR"
    local release="${base}/releases/${plan_id}"
    if ssh "$REMOTE" "test -d ${release}"; then
        echo "Reusing frozen release ${plan_id}"
    else
        local staging="${base}/releases/.upload-${plan_id}-$$"
        ssh "$REMOTE" "mkdir -p ${staging} ${base}/training-results ${base}/benchmark-results"
        rsync -az --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
            --exclude='training-results' --exclude='benchmark-results' --exclude='logs' \
            "${PROJECT_ROOT}/" "${REMOTE}:${staging}/"
        rsync -az "$plan" "${REMOTE}:${staging}/run-plan.json"
        ssh "$REMOTE" "ln -s ${base}/training-results ${staging}/training-results && \
            ln -s ${base}/benchmark-results ${staging}/benchmark-results && \
            mv -T ${staging} ${release}"
    fi
    REMOTE_DIR="$release"
}

submit_job() {
    # $1 = slurm filename, rest = script args
    local script="$1"; shift || true
    local args
    printf -v args '%q ' "$@"
    local out
    out=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/${script} ${args}")
    echo "$out"
    local job_id
    job_id=$(echo "$out" | awk '{print $NF}')
    echo "  log: ssh ${REMOTE} 'tail -f ${REMOTE_DIR}/logs/*${job_id}*.out'"
}

pull_results() {
    mkdir -p "${PROJECT_ROOT}/training-results" "${PROJECT_ROOT}/benchmark-results"
    echo "Pulling training-results..."
    rsync -az "${REMOTE}:${REMOTE_DIR}/training-results/" "${PROJECT_ROOT}/training-results/" || true
    echo "Pulling benchmark-results..."
    rsync -az "${REMOTE}:${REMOTE_DIR}/benchmark-results/" "${PROJECT_ROOT}/benchmark-results/" || true
    echo "Pull complete."
}

case "$ACTION" in
    setup)
        check_ssh; sync_code
        ssh -t "$REMOTE" "cd ${REMOTE_DIR} && CONDA_ENV=${CONDA_ENV} bash idun/setup_env.sh"
        ;;

    sync)
        check_ssh; sync_code
        ;;

    search)
        check_ssh; sync_code
        shift || true
        submit_job job_search.slurm "$@"
        ;;

    profile)
        check_ssh; sync_code
        shift || true
        submit_job job_profile.slurm "$@"
        ;;

    train)
        shift || true
        job_config="${1:-configs/ppo-gru.json}"
        job_name="${2:-train}"
        shift 2 2>/dev/null || true
        plan=""
        extras=()
        while (( $# )); do
            if [[ "$1" == "--run-plan" ]]; then
                [[ $# -ge 2 ]] || { echo "--run-plan needs a path"; exit 1; }
                plan="$2"; shift 2
            else
                extras+=("$1"); shift
            fi
        done
        [[ -f "$plan" ]] || { echo "Training requires a user-reviewed --run-plan file; nothing submitted."; exit 1; }
        check_ssh
        sync_training_release "$plan"
        submit_job job_train.slurm "$job_config" "$job_name" --run-plan run-plan.json "${extras[@]}"
        ;;

    benchmark)
        check_ssh; sync_code
        shift || true
        submit_job job_benchmark.slurm "$@"
        ;;

    compare)
        check_ssh; sync_code
        shift || true
        submit_job job_benchmark.slurm compare "$@"
        ;;

    test)
        check_ssh; sync_code
        submit_job job_test.slurm
        ;;

    serve)
        check_ssh; sync_code
        shift || true
        job_policy="${1:-configs/controller.json}"
        submit_job job_serve.slurm "${job_policy}"
        ;;

    pull)
        check_ssh; pull_results
        ;;

    queue)
        check_ssh
        ssh "$REMOTE" "squeue -u \$(whoami) --format='%.10i %.12P %.22j %.2t %.10M %.6D %R'"
        ;;

    logs)
        check_ssh
        target="${2:-}"
        if [ -n "$target" ]; then
            ssh "$REMOTE" "tail -n 100 -f ${REMOTE_DIR}/logs/*${target}*.out"
        else
            latest=$(ssh "$REMOTE" "ls -t ${REMOTE_DIR}/logs/*.out 2>/dev/null | head -n 1" || true)
            [ -z "$latest" ] && { echo "No logs found."; exit 0; }
            echo "Tailing ${latest}"
            ssh "$REMOTE" "tail -n 100 -f ${latest}"
        fi
        ;;

    help|--help|-h)
        sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        ;;

    *)
        echo "Unknown action: '$ACTION'"; exit 1
        ;;
esac
