#!/usr/bin/env bash
# Build and run the CPU image on the Azure VM, straight on its public IP (no tunnel).
#
#   bash deploy_vm.sh <vm-ip> [weights.pt | <name>_openvino_model/] [KEY=VALUE ...]
#   bash deploy_vm.sh 9.160.106.215 runs/x/weights/last_int8_openvino_model DRONE_SMALL_IMGSZ=1600
#
# Settings after the weights go into the container's environment (solution.py reads them).
# An OpenVINO folder (ultralytics export) is served as it is: ~1.9x faster on this CPU in int8.
#
# The container restarts with the VM (--restart unless-stopped): after `az vm start` the
# service is back on http://<ip>:9053/predict within a minute, no redeploy needed.
# ~/drone/logs on the VM is mounted at /logs and survives redeploys: DRONE_LOG_RESPONSES=/logs/responses.jsonl.
# This only serves. Starting a validation run happens in the competition website, by a person.
set -euo pipefail
cd "$(dirname "$0")"
IP="${1:?usage: deploy_vm.sh <vm-ip> [weights.pt | openvino folder] [KEY=VALUE ...]}"
WEIGHTS="${2:-runs/synth400_padded_0919-0824/weights/best.pt}"
shift $(( $# < 2 ? $# : 2 ))
ENVS=""
for kv in "$@"; do ENVS="$ENVS -e $kv"; done
if [ -d "$WEIGHTS" ]; then
    SERVED=model_openvino_model; ENVS="$ENVS -e DRONE_MODEL=/app/weights/$SERVED"
else
    [ -s "$WEIGHTS" ] || { echo "no weights at $WEIGHTS"; exit 1; }
    SERVED=best.pt
fi
SSH="ssh -o ConnectTimeout=10 azureuser@$IP"

echo "copying code and $WEIGHTS"
$SSH 'rm -rf ~/drone/serve_weights && mkdir -p ~/drone/serve_weights'
rsync -az --delete src requirements.txt Dockerfile.cpu "azureuser@$IP:drone/"
if [ -d "$WEIGHTS" ]; then SRC="${WEIGHTS%/}/"; else SRC="$WEIGHTS"; fi   # a folder: its contents, not nested
rsync -az "$SRC" "azureuser@$IP:drone/serve_weights/$SERVED"
echo "settings:${ENVS:- none}"

echo "building the image on the VM (the first build downloads torch: a few minutes)"
$SSH 'cd ~/drone && docker build -q -f Dockerfile.cpu -t drone-cpu . \
      && (docker rm -f drone >/dev/null 2>&1 || true) \
      && mkdir -p ~/drone/logs \
      && docker run -d --name drone --restart unless-stopped -p 9053:9053 -v ~/drone/logs:/logs'"$ENVS"' drone-cpu >/dev/null'

printf "waiting for the model to load"
for _ in $(seq 60); do
    if $SSH 'docker logs drone 2>&1' | grep -q "Loaded "; then echo " - loaded"; break; fi
    printf "."; sleep 3
done
curl -s -o /dev/null -m 10 -w "docs from here: HTTP %{http_code}\n" "http://$IP:9053/docs" || true
echo
echo "Submit this URL in the drone form (Verify first):"
echo "    http://$IP:9053/predict"
