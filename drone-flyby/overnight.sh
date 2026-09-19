#!/bin/bash
# Score every IDUN checkpoint against the Copenhagen labels, unattended, one replay at a time.
#
#   setsid nohup ~/Desktop/nordic-ai-cup-2026-amigos/drone-flyby/overnight.sh > /dev/null 2>&1 &
#
# Detached on purpose: it keeps running when VS Code, the terminal or the chat closes.
# Results, best first:   sort -k3 -rn ~/Desktop/nordic-ai-cup-2026-amigos/drone-flyby/overnight/scores.tsv
# Stop it:               cat ~/Desktop/.../overnight/pid && kill <pid>
#
# Pass 1 scores epoch15 and last of every run, so every recipe is measured early.
# Pass 2 fills in the remaining epochs, best runs first.
# The bar is the live model: 0.640 overall, 0.636 on frames 1-125.

cd "$(dirname "$0")" || exit 1
ROOT=$PWD
OUT=$ROOT/overnight
mkdir -p "$OUT"
echo $$ > "$OUT/pid"
TSV=$OUT/scores.tsv
LOG=$OUT/log.txt
touch "$TSV"
exec >> "$LOG" 2>&1
echo "=== started $(date -Is) ==="

# Every yolo11s run of this push, including the three launched after the ground-mask finding
# (trimsng, ctrl, dkplus) and the live model itself, which should come back at 0.640.
RUNS_GLOB='synth400_11s_*'

score_one() {   # $1 run, $2 epoch tag
  local run=$1 e=$2
  grep -qP "^\Q$run\E\t\Q$e\E\t" "$TSV" && return 0
  ssh -o BatchMode=yes idun "test -f ~/nordic-cup/drone-flyby/runs/$run/weights/$e.pt" || return 1
  mkdir -p "runs/$run/weights"
  rsync -a idun:nordic-cup/drone-flyby/runs/$run/weights/$e.pt "runs/$run/weights/" || return 1
  echo "--- $(date +%H:%M) scoring $run $e"
  if [ ! -d "runs/$run/weights/${e}_int8_openvino_model" ]; then
    .venv/bin/python -c "
from ultralytics import YOLO
YOLO('runs/$run/weights/$e.pt').export(format='openvino', dynamic=True, int8=True, imgsz=960,
                                       data='$ROOT/datasets/helsinki_yolo/data.yaml', fraction=1.0)" || return 1
  fi
  local res
  res=$(timeout 1800 .venv/bin/python training/run_policy.py --scene validation_4k --camera sweep --lag 1 \
        --model "runs/$run/weights/${e}_int8_openvino_model" \
        --set DRONE_SMALL_IMGSZ=1280 V2_TRUNC=1 DRONE_MEMORY_LEAD=0.1 --name "ov_${run}_$e" 2>&1)
  local all tune check
  all=$(sed -n 's/.*COCO mAP@0.50: \([0-9.]*\).*/\1/p' <<< "$res" | head -1)
  tune=$(sed -n 's/.*(tune) \([0-9.]*\).*/\1/p' <<< "$res" | head -1)
  check=$(sed -n 's/.*(check) \([0-9.]*\).*/\1/p' <<< "$res" | head -1)
  [ -z "$all" ] && { echo "    FAILED"; tail -5 <<< "$res"; return 1; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$run" "$e" "$all" "${tune:-?}" "${check:-?}" >> "$TSV"
  echo "    $run $e -> $all (tune $tune, check $check)"
  # Anything past the live model gets a note of its own, so it is easy to spot in the morning.
  if awk -v a="$all" -v t="${tune:-0}" 'BEGIN{exit !(a>0.640 && t>0.636)}'; then
    printf '%s %s  overall %s  tune %s  check %s\n' "$run" "$e" "$all" "$tune" "$check" >> "$OUT/BEATS_LIVE.txt"
    return 0   # keep the weights and the export: this one is a candidate to deploy
  fi
  # Keep the disk in check: the .pt and the export are ~40 MB each. Still on IDUN if wanted again.
  rm -rf "runs/$run/weights/${e}_int8_openvino_model" "runs/$run/weights/$e.pt"
}

runs_list() {
  ssh -o BatchMode=yes idun "cd ~/nordic-cup/drone-flyby/runs && ls -d $RUNS_GLOB 2>/dev/null"
}

for round in $(seq 200); do
  mapfile -t runs < <(runs_list)
  [ ${#runs[@]} -eq 0 ] && { sleep 300; continue; }
  progressed=0
  # Pass 1: the two checkpoints that tell us most about a recipe.
  for run in "${runs[@]}"; do
    for e in epoch15 last; do score_one "$run" "$e" && progressed=1; done
  done
  # Pass 2: the rest, best runs first.
  while read -r run; do
    for e in epoch10 epoch20 epoch25 epoch5; do score_one "$run" "$e" && progressed=1; done
  done < <(sort -k3 -rn "$TSV" | cut -f1 | awk '!seen[$0]++')
  [ $progressed = 0 ] && sleep 420
  # Everything scored and nothing left training? Then we are done.
  if [ "$(wc -l < "$TSV")" -ge 40 ] && ! ssh -o BatchMode=yes idun 'squeue -u $USER -h' | grep -q .; then
    break
  fi
done
echo "=== finished $(date -Is) ==="
echo finished >> "$TSV"
