#!/bin/bash
# Start / stop the medical-appointment endpoint and its public tunnel, and
# report the URL. Designed for the GTX 1650 (4 GB) serving config.
#
#   bash local/serve.sh start     # server + cloudflared quick tunnel
#   bash local/serve.sh url       # print the current public /predict URL
#   bash local/serve.sh status    # server/tunnel/GPU
#   bash local/serve.sh stop      # stop both
#
# No Cloudflare domain today, so this uses an ephemeral quick tunnel; the URL
# changes on restart. See docs/local-serving.md for the named-tunnel variant.

set -uo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${LOGDIR:-/tmp/opencode}"
mkdir -p "$LOGDIR"

export WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"
export WHISPER_COMPUTE_TYPE="${WHISPER_COMPUTE_TYPE:-int8}"
export NLI_DEVICE="${NLI_DEVICE:-cuda}"
export MEDAPP_NLI_TAU="${MEDAPP_NLI_TAU:-0.3}"
export MEDAPP_DECISION_NEIGHBOURS="${MEDAPP_DECISION_NEIGHBOURS:-1}"
export MEDAPP_TOP_CLAUSES="${MEDAPP_TOP_CLAUSES:-1}"
export MEDAPP_SELECT="${MEDAPP_SELECT:-greedy_trim}"
export MEDAPP_TRIM_MARGIN="${MEDAPP_TRIM_MARGIN:-0.1}"
export MEDAPP_MAX_CANDIDATES="${MEDAPP_MAX_CANDIDATES:-120}"
export MEDAPP_MAX_RANGE_WORDS="${MEDAPP_MAX_RANGE_WORDS:-10}"
export MEDAPP_DEADLINE_S="${MEDAPP_DEADLINE_S:-50}"
export MEDAPP_CAPTURE="${MEDAPP_CAPTURE:-1}"

server_pid() { pgrep -f "[p]ython api.py" || true; }
tunnel_pid() { pgrep -f "[c]loudflared tunnel --url http://localhost:9054" || true; }

start() {
    if [ -n "$(server_pid)" ]; then
        echo "server already running (pid $(server_pid))"
    else
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda activate medapp-local
        cd "$PROJECT"
        setsid nohup python api.py > "$LOGDIR/api_local.log" 2>&1 < /dev/null &
        disown
        echo "server starting; waiting for warm-up..."
        for _ in $(seq 1 40); do
            curl -s --max-time 3 http://localhost:9054/ >/dev/null 2>&1 && break
            sleep 2
        done
        curl -s --max-time 3 http://localhost:9054/ && echo
    fi

    if [ -n "$(tunnel_pid)" ]; then
        echo "tunnel already running (pid $(tunnel_pid))"
    else
        setsid nohup "$HOME/.local/bin/cloudflared" tunnel \
            --url http://localhost:9054 --no-autoupdate \
            > "$LOGDIR/cloudflared.log" 2>&1 < /dev/null &
        disown
        echo "tunnel starting; waiting for a URL..."
        for _ in $(seq 1 30); do
            [ -n "$(url)" ] && break
            sleep 2
        done
    fi
    url
}

url() {
    local u
    u=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOGDIR/cloudflared.log" 2>/dev/null | tail -1)
    [ -n "$u" ] && echo "$u/predict" || echo "(no tunnel URL yet)"
}

status() {
    echo "server: $(server_pid | paste -sd, -)"
    echo "tunnel: $(tunnel_pid | paste -sd, -)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
}

stop() {
    [ -n "$(server_pid)" ] && kill $(server_pid) 2>/dev/null
    [ -n "$(tunnel_pid)" ] && kill $(tunnel_pid) 2>/dev/null
    pkill -f "[p]ython api.py" 2>/dev/null || true
    pkill -f "[c]loudflared tunnel --url http://localhost:9054" 2>/dev/null || true
    sleep 1
    echo "stopped. server: '$([ -z "$(server_pid)" ] && echo none)' tunnel: '$([ -z "$(tunnel_pid)" ] && echo none)'"
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    url) url ;;
    status) status ;;
    *) echo "usage: $0 {start|stop|url|status}"; exit 1 ;;
esac
