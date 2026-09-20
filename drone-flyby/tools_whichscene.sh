#!/bin/bash
# Tag the most recent evaluator run with the scene it received.
# Compares ONLY views with the SAME filename (frame + level + camera centre): different
# models make different camera decisions, so comparing frame 2 against frame 1 shows a
# large difference even on an identical scene. That bug once cried "a third scene".
SP=/tmp/claude-1000/-home-juan-Desktop-nordic-ai-cup-2026-amigos/e8339658-3589-40b4-957e-6388df2e64fd/scratchpad
VM=azureuser@9.160.106.215
S=$(ssh -o ConnectTimeout=10 -o BatchMode=yes $VM 'docker exec drone sh -c "ls -t /recordings | head -1"' 2>/dev/null | tr -d '\r')
N=$(ssh -o ConnectTimeout=10 -o BatchMode=yes $VM "docker exec drone sh -c 'ls /recordings/$S | wc -l'" 2>/dev/null | tr -d '\r')
rm -rf "$SP/_last"; mkdir -p "$SP/_last"
ssh -o ConnectTimeout=10 -o BatchMode=yes $VM \
  "docker cp drone:/recordings/$S /tmp/_q >/dev/null 2>&1; tar -cf - -C /tmp/_q \$(ls /tmp/_q | grep L0 | head -4); rm -rf /tmp/_q" 2>/dev/null \
  | tar -xf - -C "$SP/_last" 2>/dev/null
cd ~/Desktop/nordic-ai-cup-2026-amigos/drone-flyby
.venv/bin/python - "$S" "$N" <<'PY'
import cv2, numpy as np, os, sys
SP='/tmp/claude-1000/-home-juan-Desktop-nordic-ai-cup-2026-amigos/e8339658-3589-40b4-957e-6388df2e64fd/scratchpad'
got=sorted(os.listdir(SP+'/_last'))
best=(None, 1e9, None)
for scene in ('old','forest'):
    for name in got:
        ref=f'{SP}/refs/{scene}/{name}'
        if not os.path.exists(ref): continue
        a=cv2.imread(f'{SP}/_last/{name}'); b=cv2.imread(ref)
        if a is None or b is None or a.shape!=b.shape: continue
        d=float(np.abs(a.astype(np.int16)-b.astype(np.int16)).mean())
        if d<best[1]: best=(scene,d,name)
scene,d,name=best
if scene is None:
    print(f'run {sys.argv[1][:12]}  {sys.argv[2]} files  NO COMPARABLE FRAME (views: {got})')
else:
    verdict=scene.upper() if d<1.0 else f'UNKNOWN - closest {scene} at {d:.1f}, possibly a NEW scene'
    print(f'run {sys.argv[1][:12]}  {sys.argv[2]} files  scene = {verdict}  (matched on {name}, diff {d:.2f})')
PY
