#!/usr/bin/env bash
# Verify the VM is serving exactly the model we intend, before pressing submit.
#
#   bash preflight.sh runs/synth400_11s_bigbg_0920-0209/weights/epoch15_int8_openvino_model
#
# Checks, in order: container up, weights md5 match the local export, serving settings,
# and one real 4K frame answered end to end inside the 3333 ms budget.
# The smoke test uses its own sequence_id; solution._states is keyed by it, so it cannot
# touch a real run's tracking memory.
set -uo pipefail
cd "$(dirname "$0")"
IP=${IP:-9.160.106.215}
WANT=${1:?usage: bash preflight.sh <local _int8_openvino_model dir>}
SSH="ssh -o ConnectTimeout=10 -o BatchMode=yes azureuser@$IP"
fail=0
ok(){ printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== container =="
st=$($SSH 'docker ps --filter name=drone --format "{{.Status}}"' 2>/dev/null)
[ -n "$st" ] && ok "drone $st" || bad "container not running"

echo "== weights =="
remote=$($SSH 'md5sum ~/drone/serve_weights/model_openvino_model/*.bin' 2>/dev/null | awk '{print $1}')
local_=$(md5sum "$WANT"/*.bin 2>/dev/null | awk '{print $1}')
if [ -n "$remote" ] && [ "$remote" = "$local_" ]; then ok "md5 $remote matches $WANT"
else bad "served $remote != local $local_ ($WANT)"; fi

echo "== settings =="
env=$($SSH 'docker inspect drone --format "{{range .Config.Env}}{{println .}}{{end}}"' 2>/dev/null | grep -E '^(DRONE_|V2_)' | sort | tr '\n' ' ')
echo "  $env"
for want in DRONE_SMALL_IMGSZ=1280 V2_TRUNC=1 DRONE_MEMORY_LEAD=0.1 DRONE_CAMERA=sweep; do
  case "$env" in *"$want"*) ;; *) bad "missing $want" ;; esac
done
[ $fail = 0 ] && ok "serving settings as expected"

echo "== end to end =="
.venv/bin/python - "$IP" <<'PY'
import base64, json, sys, time, urllib.request
import cv2
img = cv2.imread('recordings/validation_4k/frame_0001.jpg')
b64 = base64.b64encode(cv2.imencode('.png', cv2.resize(img, (960, 540)))[1].tobytes()).decode()
req = {"sequence_id": f"preflight-{int(time.time())}", "frame": 1, "frame_index": 1,
       "request_id": "pf-1", "frame_interval_ms": 333, "response_timeout_ms": 3333,
       "original_width": 3840, "original_height": 2160,
       "view": {"resolution_level": 0, "center_x": 1920, "center_y": 1080, "view_id": "v1",
                "image": b64, "image_media_type": "image/png", "width": 960, "height": 540,
                "source_region_xyxy": [0, 0, 3840, 2160]},
       "camera_constraints": {"maximum_center_delta": 1000.0, "allowed_resolution_levels": [0,1,2],
                              "center_bounds": [], "full_view_reset_exempt_from_delta": True}}
t = time.time()
try:
    r = urllib.request.Request(f"http://{sys.argv[1]}:9053/predict", data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        out = json.loads(resp.read())
except Exception as exc:
    print(f'  \033[31mFAIL\033[0m request failed: {exc}'); sys.exit(1)
ms = (time.time() - t) * 1000
n = len(out.get('annotations', []))
verdict = '\033[32mOK\033[0m  ' if (ms < 3333 and n > 0) else '\033[31mFAIL\033[0m'
print(f'  {verdict} HTTP 200 in {ms:.0f} ms (budget 3333), {n} detections, '
      f'echoed frame={out.get("frame")} request_id={out.get("request_id")}')
sys.exit(0 if (ms < 3333 and n > 0) else 1)
PY
[ $? -ne 0 ] && fail=1
echo
[ $fail = 0 ] && echo "READY TO SUBMIT" || echo "NOT READY - fix the FAIL lines above"
exit $fail
