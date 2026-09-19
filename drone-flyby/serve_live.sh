#!/usr/bin/env bash
# Serve the solution to the evaluation service from this laptop: API server + a fresh
# cloudflared quick tunnel, then print the URL to paste in the drone form.
#
#   bash serve_live.sh                      # sweep camera, the default weights below
#   DRONE_MODEL=runs/<run>/weights/best.pt DRONE_CAMERA=hybrid bash serve_live.sh
#   bash serve_live.sh stop                 # stop both
#
# This only serves. Verifying and starting a run happen in the competition website, by a person.
# Quick tunnels get a new URL every start and die with the process; logs go to logs/live/.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-9056}"
LOGS=logs/live
mkdir -p "$LOGS"

stop() {
    for name in tunnel api; do
        if [ -f "$LOGS/$name.pid" ] && kill "$(cat "$LOGS/$name.pid")" 2>/dev/null; then
            echo "stopped $name (pid $(cat "$LOGS/$name.pid"))"
        fi
        rm -f "$LOGS/$name.pid"
    done
}

if [ "${1:-}" = stop ]; then
    stop
    exit 0
fi

export DRONE_CAMERA="${DRONE_CAMERA:-sweep}"
export DRONE_MODEL="${DRONE_MODEL:-runs/synth400_padded_0919-0824/weights/best.pt}"
DRONE_MODEL="$(realpath "$DRONE_MODEL")"
[ -s "$DRONE_MODEL" ] || { echo "no weights at $DRONE_MODEL"; exit 1; }

stop >/dev/null
if fuser "$PORT/tcp" >/dev/null 2>&1; then
    echo "port $PORT is taken by another process (fuser $PORT/tcp); stop it or set PORT"
    exit 1
fi

echo "API: camera $DRONE_CAMERA, weights $DRONE_MODEL, port $PORT"
nohup .venv/bin/python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port "$PORT" > "$LOGS/api.log" 2>&1 &
echo $! > "$LOGS/api.pid"
for _ in $(seq 90); do
    grep -q "Application startup complete" "$LOGS/api.log" && break
    kill -0 "$(cat "$LOGS/api.pid")" 2>/dev/null || { echo "API died:"; tail -20 "$LOGS/api.log"; exit 1; }
    sleep 1
done
grep -q "Loaded " "$LOGS/api.log" || { echo "API up but no model loaded:"; tail -20 "$LOGS/api.log"; stop; exit 1; }

nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" > "$LOGS/tunnel.log" 2>&1 &
echo $! > "$LOGS/tunnel.pid"
URL=""
for _ in $(seq 60); do
    URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOGS/tunnel.log" | head -1 || true)"
    [ -n "$URL" ] && break
    sleep 1
done
[ -n "$URL" ] || { echo "no tunnel URL after 60 s:"; tail -20 "$LOGS/tunnel.log"; stop; exit 1; }

echo
echo "Submit this URL in the drone form (Verify first):"
echo "    $URL/predict"
echo
# A new quick tunnel takes a little while to be reachable: wait before pressing Verify.
printf "waiting until it answers"
for _ in $(seq 40); do
    if [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$URL/docs")" = 200 ]; then echo " - reachable, go ahead"; break; fi
    printf "."
    sleep 3
done
echo
echo "Logs: $LOGS/api.log, $LOGS/tunnel.log.  Stop: bash serve_live.sh stop"
