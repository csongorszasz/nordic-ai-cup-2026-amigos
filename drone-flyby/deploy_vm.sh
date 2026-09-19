#!/usr/bin/env bash
# Build and run the CPU image on the Azure VM, straight on its public IP (no tunnel).
#
#   bash deploy_vm.sh <vm-ip> [weights.pt]
#
# The container restarts with the VM (--restart unless-stopped): after `az vm start` the
# service is back on http://<ip>:9053/predict within a minute, no redeploy needed.
# This only serves. Starting a validation run happens in the competition website, by a person.
set -euo pipefail
cd "$(dirname "$0")"
IP="${1:?usage: deploy_vm.sh <vm-ip> [weights.pt]}"
WEIGHTS="${2:-runs/synth400_padded_0919-0824/weights/best.pt}"
[ -s "$WEIGHTS" ] || { echo "no weights at $WEIGHTS"; exit 1; }
SSH="ssh -o ConnectTimeout=10 azureuser@$IP"

echo "copying code and $WEIGHTS"
$SSH 'mkdir -p ~/drone/serve_weights'
rsync -az --delete src requirements.txt Dockerfile.cpu "azureuser@$IP:drone/"
rsync -az "$WEIGHTS" "azureuser@$IP:drone/serve_weights/best.pt"

echo "building the image on the VM (the first build downloads torch: a few minutes)"
$SSH 'cd ~/drone && docker build -q -f Dockerfile.cpu -t drone-cpu . \
      && (docker rm -f drone >/dev/null 2>&1 || true) \
      && docker run -d --name drone --restart unless-stopped -p 9053:9053 drone-cpu >/dev/null'

printf "waiting for the model to load"
for _ in $(seq 60); do
    if $SSH 'docker logs drone 2>&1' | grep -q "Loaded "; then echo " - loaded"; break; fi
    printf "."; sleep 3
done
curl -s -o /dev/null -m 10 -w "docs from here: HTTP %{http_code}\n" "http://$IP:9053/docs" || true
echo
echo "Submit this URL in the drone form (Verify first):"
echo "    http://$IP:9053/predict"
