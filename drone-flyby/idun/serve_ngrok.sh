#!/usr/bin/env bash
# Serve from an IDUN GPU node through an ngrok tunnel: a trial against the Azure VM.
# Run it yourself (Claude does not open tunnels); the ngrok token stays on IDUN.
#
#   bash idun/serve_ngrok.sh          # hold a GPU for 2 days (idun/hold_node.slurm) if not yet,
#                                     # then start API + tunnel in tmux on it and show the URL
#   bash idun/serve_ngrok.sh attach   # back into the tmux session (Ctrl+B then D leaves it running)
#   bash idun/serve_ngrok.sh stop     # stop the server and free the GPU
#
# One time, on IDUN (ssh idun):  ~/bin/ngrok config add-authtoken <your token>
# Before any validation, time it against the VM from the same laptop:
#   .venv/bin/python src/local_evaluator.py --url <ngrok url>/predict --scene helsinki --realtime
#   .venv/bin/python src/local_evaluator.py --url http://<vm ip>:9053/predict --scene helsinki --realtime
# and compare "frames skipped" and "round trip ms". A tunnel lost frames this morning.
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE_DIR="${REMOTE_DIR:-~/nordic-cup/drone-flyby}"

job() { ssh idun "squeue -u \$(whoami) -n drone-serve -h -o '%i %T %N'" | head -1; }

if [ "${1:-}" = stop ]; then
    read -r id _ node <<<"$(job)" || true
    [ -n "${id:-}" ] || { echo "no serving job"; exit 0; }
    ssh idun "ssh $node 'tmux kill-session -t drone' 2>/dev/null; scancel $id" && echo "stopped job $id on $node"
    exit 0
fi

if [ "${1:-}" != attach ]; then
    bash idun/submit.sh sync
    ssh idun 'mkdir -p ~/bin && { [ -x ~/bin/ngrok ] || curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar xz -C ~/bin; } && ~/bin/ngrok version && { ~/bin/ngrok config check >/dev/null 2>&1 || echo "no ngrok token yet: ssh idun, then ~/bin/ngrok config add-authtoken <token>"; }'
    [ -n "$(job)" ] || ssh idun "cd ${REMOTE_DIR} && mkdir -p logs && sbatch idun/hold_node.slurm"
fi
printf "waiting for the GPU node"
while true; do
    read -r id state node <<<"$(job)"
    [ "${state:-}" = RUNNING ] && break
    printf "."; sleep 10
done
echo " $node (job $id)"
# Start (or rejoin) the server's tmux session on the node.
ssh -t idun "ssh -t $node 'cd ${REMOTE_DIR} && tmux new-session -A -s drone \"bash idun/serve_node.sh; echo server stopped; bash\"'"
