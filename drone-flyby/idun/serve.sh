#!/bin/bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Serving requires an owned Slurm allocation}"
cd "${SLURM_SUBMIT_DIR:?Set the immutable serving snapshot directory}"
source idun/runtime.sh

if ! python idun/preflight.py > runs/serving-preflight.log 2>&1; then
    tail -30 runs/serving-preflight.log >&2
    exit 1
fi
exec python idun/serve.py "$@"
