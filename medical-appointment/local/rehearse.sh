#!/bin/bash
# Dress rehearsal against the PUBLIC endpoint, before a validation or the
# one-shot evaluation: readiness, then all 39 training conversations replayed
# back to back through the tunnel with the service's own failure rules.
#
#   bash local/rehearse.sh https://<name>.trycloudflare.com/predict
#
# Pass criteria (checked at the end): /ready is 200, 0 timeouts, 0 failed
# requests, worst round trip under WORST_MAX_S (default 45 s).

set -uo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${1:?usage: $0 https://<host>/predict}"
BASE="${URL%/predict}"
WORST_MAX_S="${WORST_MAX_S:-45}"
LOG="${LOGDIR:-/tmp}/rehearsal_$(date +%Y%m%d_%H%M%S).log"

echo "== readiness: ${BASE}/ready"
READY_CODE=$(curl -s -o /tmp/rehearsal_ready.json -w "%{http_code}" --max-time 20 "${BASE}/ready")
cat /tmp/rehearsal_ready.json; echo
if [ "$READY_CODE" != "200" ]; then
    echo "FAIL: /ready returned ${READY_CODE}; the endpoint is not warm. Do not submit."
    exit 1
fi

echo "== replaying 39 conversations through ${URL} (log: ${LOG})"
cd "$PROJECT"
python local_evaluator.py --url "$URL" | tee "$LOG"

echo
echo "== verdict"
FAIL=0
TIMEOUTS=$(grep -aE "^  timeouts " "$LOG" | grep -oE "[0-9]+" | head -1)
FAILED=$(grep -aE "^  failed conversations " "$LOG" | grep -oE "[0-9]+" | head -1)
WORST_MS=$(grep -aE "per conversation .* ms worst" "$LOG" | grep -oE "[0-9]+ ms worst" | grep -oE "[0-9]+")
WORST=$(awk "BEGIN{printf \"%.1f\", ${WORST_MS:-0}/1000}")
echo "timeouts=${TIMEOUTS:-?} failed=${FAILED:-?} worst_round_trip=${WORST}s"
grep -aq "ABORTED" "$LOG" && { echo "FAIL: the service would have aborted the attempt"; FAIL=1; }
[ "${TIMEOUTS:-0}" != "0" ] && { echo "FAIL: timeouts"; FAIL=1; }
[ "${FAILED:-0}" != "0" ] && { echo "FAIL: failed requests"; FAIL=1; }
if [ -z "${WORST_MS:-}" ]; then
    echo "WARN: could not read the worst round trip; check the report above."
elif awk "BEGIN{exit !(${WORST} > ${WORST_MAX_S})}"; then
    echo "FAIL: worst round trip ${WORST}s > ${WORST_MAX_S}s"; FAIL=1
fi
echo "Also check the server log for 'guessing' / 'overran' / 'unanswered':"
echo "those are 200s the harness cannot tell apart from real answers."
[ "$FAIL" = "0" ] && echo "PASS: safe to validate / evaluate against ${URL}"
exit "$FAIL"
