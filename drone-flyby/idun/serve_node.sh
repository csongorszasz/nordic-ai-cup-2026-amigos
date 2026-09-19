#!/usr/bin/env bash
# On the IDUN GPU node (started by idun/serve_ngrok.sh): the API on the GPU, then an ngrok tunnel to it.
set -uo pipefail
module purge >/dev/null 2>&1
module load Anaconda3/2023.09-0 >/dev/null 2>&1
source /cluster/apps/eb/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate drone
mkdir -p logs
export DRONE_CAMERA=sweep
export DRONE_MODEL="${DRONE_MODEL:-$PWD/runs/synth400_11s_paint_0919-1333/weights/last.pt}"
export DRONE_SMALL_IMGSZ="${DRONE_SMALL_IMGSZ:-1600}"   # the GPU has time for the bigger second pass
PORT=9053
nvidia-smi --query-gpu=name --format=csv,noheader

python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port $PORT > logs/serve_api.log 2>&1 &
API=$!
trap 'kill $API ${TUNNEL:-} 2>/dev/null' EXIT
for _ in $(seq 120); do grep -q "Application startup complete" logs/serve_api.log && break; sleep 1; done
grep -q "Loaded " logs/serve_api.log || { echo "no model loaded:"; tail -20 logs/serve_api.log; exit 1; }
echo "API up: $DRONE_MODEL, second pass $DRONE_SMALL_IMGSZ"

~/bin/ngrok http $PORT --log stdout --log-format logfmt > logs/serve_ngrok.log 2>&1 &
TUNNEL=$!
URL=""
for _ in $(seq 30); do
    URL="$(grep -o 'url=https://[^ ]*' logs/serve_ngrok.log | head -1 | cut -d= -f2)"
    [ -n "$URL" ] && break
    sleep 1
done
[ -n "$URL" ] || { echo "no tunnel URL:"; tail -20 logs/serve_ngrok.log; exit 1; }
echo
echo "    $URL/predict"
echo
echo "Time it against the VM before validating (see idun/serve_ngrok.sh). Ctrl+C stops the API and tunnel."
wait $API
