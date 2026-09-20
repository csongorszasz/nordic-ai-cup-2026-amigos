#!/bin/bash
# Tag the most recent evaluator run with the scene it received.
SP=/tmp/claude-1000/-home-juan-Desktop-nordic-ai-cup-2026-amigos/e8339658-3589-40b4-957e-6388df2e64fd/scratchpad
VM=azureuser@9.160.106.215
S=$(ssh -o ConnectTimeout=10 -o BatchMode=yes $VM \
     'docker exec drone sh -c "ls -t /recordings | head -1"' 2>/dev/null | tr -d "\r")
N=$(ssh -o ConnectTimeout=10 -o BatchMode=yes $VM \
     "docker exec drone sh -c 'ls /recordings/$S | wc -l'" 2>/dev/null | tr -d "\r")
rm -rf "$SP/_last"; mkdir -p "$SP/_last"
ssh -o ConnectTimeout=10 -o BatchMode=yes $VM \
  "docker cp drone:/recordings/$S /tmp/_q >/dev/null 2>&1; F=\$(ls /tmp/_q | grep L0 | head -1); tar -cf - -C /tmp/_q \$F; rm -rf /tmp/_q" 2>/dev/null \
  | tar -xf - -C "$SP/_last" 2>/dev/null
cd ~/Desktop/nordic-ai-cup-2026-amigos/drone-flyby
.venv/bin/python - "$S" "$N" <<'PY'
import cv2, numpy as np, glob, sys
SP='/tmp/claude-1000/-home-juan-Desktop-nordic-ai-cup-2026-amigos/e8339658-3589-40b4-957e-6388df2e64fd/scratchpad'
f=glob.glob(SP+'/_last/*.png')
if not f: print('no L0 frame recorded yet'); raise SystemExit
im=cv2.imread(f[0])
best=None
for name in ('old','forest'):
    r=cv2.imread(f'{SP}/refs/{name}.png')
    d=float(np.abs(im.astype(np.int16)-r.astype(np.int16)).mean())
    if best is None or d<best[1]: best=(name,d)
tag, d = best
print(f'run {sys.argv[1][:12]}  {sys.argv[2]} files  scene = {tag.upper()}  (diff {d:.2f})'
      + ('' if d < 5 else '   <-- MATCHES NEITHER: a third scene'))
PY
