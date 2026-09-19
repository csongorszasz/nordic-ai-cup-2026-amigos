#!/usr/bin/env bash
# Serve from an IDUN GPU node through an ngrok tunnel: a trial against the Azure VM.
# Run it yourself (Claude does not open tunnels); the ngrok token stays on IDUN.
#
#   bash idun/serve_ngrok.sh [hours]        # from drone-flyby/ on the laptop; Ctrl+C ends it all
#
# One time, on IDUN (ssh idun):  ~/bin/ngrok config add-authtoken <your token>
# Before any validation, time it against the VM from the same laptop:
#   .venv/bin/python src/local_evaluator.py --url <ngrok url>/predict --scene helsinki --realtime
#   .venv/bin/python src/local_evaluator.py --url http://<vm ip>:9053/predict --scene helsinki --realtime
# and look at "frames skipped" and "round trip ms". The tunnel lost frames this morning.
set -euo pipefail
cd "$(dirname "$0")/.."
HOURS="${1:-12}"
REMOTE_DIR="${REMOTE_DIR:-~/nordic-cup/drone-flyby}"

bash idun/submit.sh sync
ssh idun 'mkdir -p ~/bin && { [ -x ~/bin/ngrok ] || curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C ~/bin; } && ~/bin/ngrok version && { ~/bin/ngrok config check >/dev/null 2>&1 || echo "no ngrok token yet: ssh idun, then ~/bin/ngrok config add-authtoken <token>"; }'
echo "asking for a GPU node for ${HOURS} h (it may wait in the queue)"
ssh -t idun "cd ${REMOTE_DIR} && srun --account=share-ie-idi --partition=GPUQ --gres=gpu:1 --constraint='gpu32g|gpu40g|gpu80g' --cpus-per-task=8 --mem=32G --time=${HOURS}:00:00 --pty bash idun/serve_node.sh"
