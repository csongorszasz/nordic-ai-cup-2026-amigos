#!/bin/bash
# End-to-end tunnel check: a trivial HTTP origin on the compute node, exposed
# through a cloudflared quick tunnel, then fetched back through the public URL.
#
#   bash idun/net_e2e_check.sh
set -uo pipefail

CF="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
CF_LOG="${LOGDIR:-/tmp}/cf_e2e.log"
HTTP_LOG="${LOGDIR:-/tmp}/http_e2e.log"

echo "host: $(hostname)   $(date)"
python -m http.server 9054 --bind 127.0.0.1 > "$HTTP_LOG" 2>&1 &
srv=$!
sleep 2

"$CF" tunnel --url http://localhost:9054 --no-autoupdate > "$CF_LOG" 2>&1 &
cf=$!

url=""
for _ in $(seq 1 30); do
    url=$(grep -aoE "https://[a-z0-9-]+\.trycloudflare\.com" "$CF_LOG" 2>/dev/null | tail -1)
    [ -n "$url" ] && break
    sleep 2
done
echo "PUBLIC_URL: ${url:-NONE}"
sleep 6

echo "--- round-trip through Cloudflare ---"
for _ in 1 2 3; do
    curl -s -o /dev/null -w "  GET / -> %{http_code} in %{time_total}s\n" \
        --max-time 20 "$url/" 2>&1 || echo "  curl failed"
    sleep 3
done

# Keep the tunnel alive briefly so the caller can also fetch it from outside.
if [ -n "$url" ]; then
    echo "HOLDING_URL for 90s: $url"
    sleep 90
fi

kill "$cf" "$srv" 2>/dev/null || true
echo "done $(date)"
