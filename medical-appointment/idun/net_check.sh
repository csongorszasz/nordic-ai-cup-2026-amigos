#!/bin/bash
# Quick compute-node network check: outbound HTTPS + cloudflared quick tunnel.
#
# Run on a GPU node (see job_net_test.slurm). cloudflared must already be staged
# on the login node (~/.local/bin/cloudflared) so this isolates compute egress.
#
#   bash idun/net_check.sh
set -uo pipefail

CF="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
LOG="${LOGDIR:-/tmp}/cf_net_test.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

echo "host: $(hostname)   $(date)"
echo "--- outbound HTTPS ---"
for url in https://www.cloudflare.com https://github.com https://api.cloudflare.com; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$url" 2>/dev/null || echo ERR)
    echo "  $url -> $code"
done

echo "--- DNS ---"
getent hosts api.cloudflare.com >/dev/null 2>&1 && echo "  api.cloudflare.com resolves" || echo "  DNS fail"

echo "--- cloudflared ---"
if [ ! -x "$CF" ]; then
    echo "  cloudflared not found/executable at $CF"
    exit 0
fi
"$CF" --version 2>&1 | head -1

"$CF" tunnel --url http://localhost:9054 --no-autoupdate > "$LOG" 2>&1 &
pid=$!
url=""
for _ in $(seq 1 20); do
    url=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG" 2>/dev/null | tail -1)
    [ -n "$url" ] && break
    sleep 2
done
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true

echo "  quick-tunnel url: ${url:-NONE}"
echo "--- cloudflared log (tail) ---"
tail -n 20 "$LOG" 2>/dev/null || true
[ -n "$url" ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
