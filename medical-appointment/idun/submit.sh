#!/bin/bash
# ==============================================================================
# IDUN deployment & SLURM submission harness for medical-appointment
# ==============================================================================
#
# Usage (from the medical-appointment project root):
#   bash idun/submit.sh sync        # rsync project (incl. data/) to IDUN
#   bash idun/submit.sh setup       # create/update the `nordic` conda env + weights
#   bash idun/submit.sh dev [args]  # fast test gate + in-process dev_eval on GPU
#   bash idun/submit.sh test        # slow (model-backed) test suite on GPU
#   bash idun/submit.sh train [args]# train the ModernBERT answerer (grouped OOF)
#   bash idun/submit.sh refiner [args]# train the post-hoc boundary refiner
#   bash idun/submit.sh llm [args]  # LLM ceiling probe (L0/L1/L2)
#   bash idun/submit.sh llm80 [args]# same, forced onto an 80 GB GPU (26B/31B)
#   bash idun/submit.sh serve       # serve the LLM endpoint + cloudflared tunnel
#   bash idun/submit.sh eval        # submit an end-to-end HTTP scoring job
#   bash idun/submit.sh queue       # show your SLURM jobs
#   bash idun/submit.sh logs [id]   # tail a job log
#   bash idun/submit.sh pull        # copy results/ and transcripts/ back locally
#
# Environment overrides:
#   REMOTE="idun"                  SSH alias from ~/.ssh/config
#   REMOTE_DIR="~/nordic-medical"  target dir on IDUN
#   SLURM_ACCOUNT="share-ie-idi"   SLURM allocation
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-idun}"
REMOTE_DIR="${REMOTE_DIR:-~/nordic-medical}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-share-ie-idi}"

ACTION="${1:-help}"

check_ssh() {
    echo -n "Checking SSH connection to '${REMOTE}'... "
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" "echo ok" >/dev/null 2>&1; then
        echo "FAILED"
        echo "  Cannot connect. IDUN requires NTNU VPN/eduroam and an SSH key."
        exit 1
    fi
    echo "OK"
}

sync_code() {
    echo "Syncing ${PROJECT_ROOT} -> ${REMOTE}:${REMOTE_DIR}"
    ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}"
    rsync -avz --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='venv' \
        --exclude='models' \
        --exclude='transcripts' \
        --exclude='logs' \
        --exclude='results' \
        --exclude='.scratch' \
        "${PROJECT_ROOT}/" "${REMOTE}:${REMOTE_DIR}/"
    echo "Sync complete."
}

# Training reads cached transcripts directly; `sync` excludes transcripts/ so
# remote caches are never deleted, so push the local ones explicitly.
sync_transcripts() {
    echo "Syncing transcripts -> ${REMOTE}:${REMOTE_DIR}/transcripts"
    ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}/transcripts"
    rsync -avz "${PROJECT_ROOT}/transcripts/" "${REMOTE}:${REMOTE_DIR}/transcripts/"
}

# Copy experiment summaries (and transcripts) back from IDUN. results/ is
# excluded from sync so it is never deleted remotely; pull it to keep a record.
pull_results() {
    mkdir -p "${PROJECT_ROOT}/results" "${PROJECT_ROOT}/transcripts" "${PROJECT_ROOT}/models"
    echo "Pulling results..."
    rsync -avz "${REMOTE}:${REMOTE_DIR}/results/" "${PROJECT_ROOT}/results/"
    echo "Pulling transcripts..."
    rsync -avz "${REMOTE}:${REMOTE_DIR}/transcripts/" "${PROJECT_ROOT}/transcripts/"
    echo "Pulling model checkpoints..."
    rsync -avz "${REMOTE}:${REMOTE_DIR}/models/" "${PROJECT_ROOT}/models/" || true
    echo "Pull complete."
}

case "$ACTION" in
    sync)
        check_ssh; sync_code
        ;;

    setup)
        check_ssh; sync_code
        echo "Provisioning environment on IDUN (login node)..."
        ssh -t "$REMOTE" "cd ${REMOTE_DIR} && bash idun/setup_env.sh"
        ;;

    serve)
        check_ssh; sync_code; sync_transcripts
        # Forward the LLM selection and allow per-run sbatch overrides (e.g. a
        # 26B on gpu80g). Defaults keep the E4B/gpu40g config.
        ENV_ARGS=""
        [ -n "${MEDAPP_LLM_MODEL:-}" ] && ENV_ARGS="${ENV_ARGS} MEDAPP_LLM_MODEL=${MEDAPP_LLM_MODEL}"
        [ -n "${MEDAPP_LLM_PROMPT:-}" ] && ENV_ARGS="${ENV_ARGS} MEDAPP_LLM_PROMPT=${MEDAPP_LLM_PROMPT}"
        SBATCH_OPTS=""
        [ -n "${MEDAPP_SERVE_CONSTRAINT:-}" ] && SBATCH_OPTS="${SBATCH_OPTS} --constraint=${MEDAPP_SERVE_CONSTRAINT}"
        [ -n "${MEDAPP_SERVE_MEM:-}" ] && SBATCH_OPTS="${SBATCH_OPTS} --mem=${MEDAPP_SERVE_MEM}"
        [ -n "${MEDAPP_SERVE_TIME:-}" ] && SBATCH_OPTS="${SBATCH_OPTS} --time=${MEDAPP_SERVE_TIME}"
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && ${ENV_ARGS} sbatch --account=${SLURM_ACCOUNT}${SBATCH_OPTS} idun/job_serve_llm.slurm")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  URL: ssh ${REMOTE} \"grep -a PUBLIC_URL ${REMOTE_DIR}/logs/med_serve_llm_${JOB_ID}.out\""
        echo "  log: tail -f ${REMOTE_DIR}/logs/med_serve_llm_${JOB_ID}.out"
        ;;

    eval)
        check_ssh; sync_code
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_eval.slurm")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo ""
        echo "Monitor:"
        echo "  ssh ${REMOTE} 'squeue -u \$(whoami)'"
        echo "  ssh ${REMOTE} 'tail -f ${REMOTE_DIR}/logs/med_eval_${JOB_ID}.out'"
        ;;

    dev)
        check_ssh; sync_code
        shift || true
        ARGS="$*"
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_dev_eval.slurm ${ARGS}")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  tail -f ${REMOTE_DIR}/logs/med_dev_eval_${JOB_ID}.out"
        ;;

    test)
        check_ssh; sync_code
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_test.slurm")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  tail -f ${REMOTE_DIR}/logs/med_tests_${JOB_ID}.out"
        ;;

    train)
        check_ssh; sync_code; sync_transcripts
        shift || true
        ARGS="$*"
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_train_modernbert.slurm ${ARGS}")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  tail -f ${REMOTE_DIR}/logs/med_train_mb_${JOB_ID}.out"
        ;;

    refiner)
        check_ssh; sync_code; sync_transcripts
        shift || true
        ARGS="$*"
        # results/ is excluded from sync; the refiner needs the E4B probe records.
        ssh "$REMOTE" "mkdir -p ${REMOTE_DIR}/results"
        rsync -avz "${PROJECT_ROOT}/results/llm_gemma_l1_L1_questions.json" \
            "${REMOTE}:${REMOTE_DIR}/results/"
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_train_refiner.slurm ${ARGS}")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  tail -f ${REMOTE_DIR}/logs/med_train_refiner_${JOB_ID}.out"
        ;;

    llm)
        check_ssh; sync_code
        shift || true
        ARGS="$*"
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_llm_probe.slurm ${ARGS}")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  tail -f ${REMOTE_DIR}/logs/med_llm_probe_${JOB_ID}.out"
        ;;

    audio)
        check_ssh; sync_code
        shift || true
        ARGS="$*"
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_audio_probe.slurm ${ARGS}")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  tail -f ${REMOTE_DIR}/logs/med_audio_probe_${JOB_ID}.out"
        ;;

    llm80)
        check_ssh; sync_code
        shift || true
        ARGS="$*"
        JOB_SUBMIT=$(ssh "$REMOTE" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch --account=${SLURM_ACCOUNT} idun/job_llm_probe_80g.slurm ${ARGS}")
        echo "$JOB_SUBMIT"
        JOB_ID=$(echo "$JOB_SUBMIT" | awk '{print $NF}')
        echo "  tail -f ${REMOTE_DIR}/logs/med_llm_probe_80g_${JOB_ID}.out"
        ;;

    queue)
        check_ssh
        ssh "$REMOTE" "squeue -u \$(whoami) --format='%.10i %.12P %.20j %.8u %.2t %.10M %.6D %R'"
        ;;

    logs)
        check_ssh
        TARGET="${2:-}"
        if [ -n "$TARGET" ]; then
            ssh "$REMOTE" "tail -n 80 -f ${REMOTE_DIR}/logs/*${TARGET}*.out"
        else
            LATEST=$(ssh "$REMOTE" "ls -t ${REMOTE_DIR}/logs/*.out 2>/dev/null | head -n 1" || true)
            [ -z "$LATEST" ] && { echo "No logs found."; exit 0; }
            echo "Tailing ${LATEST}"
            ssh "$REMOTE" "tail -n 80 -f ${LATEST}"
        fi
        ;;

    pull)
        check_ssh
        pull_results
        ;;

    help|--help|-h)
        sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        ;;

    *)
        echo "Unknown action: '$ACTION'"; exit 1
        ;;
esac
