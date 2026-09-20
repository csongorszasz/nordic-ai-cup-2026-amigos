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
  # Selection happens on the tune half (frames 1-125) ONLY. allbg last scored 0.663 overall
  # against the live model's 0.640 and was deployed on that; live it gave 0.4926 against
  # 0.5171. Its tune half was 0.635 to the live model's 0.636 -- a tie -- so all of the
  # apparent gain sat in the confirmation half. Judging on the overall number is judging on
  # the half kept back for confirmation, and it cost us a validation.
  if awk -v t="${tune:-0}" 'BEGIN{exit !(t>0.636)}'; then
    printf '%s %s  tune %s  overall %s  check %s\n' "$run" "$e" "$tune" "$all" "$check" >> "$OUT/BEATS_LIVE.txt"
    return 0   # a real candidate: better on the half we are allowed to choose on
  fi
  if awk -v a="$all" 'BEGIN{exit !(a>0.640)}'; then
    # Good overall but not on the tune half: keep the weights, do not treat it as a candidate.
    printf '%s %s  tune %s  overall %s  check %s\n' "$run" "$e" "$tune" "$all" "$check" >> "$OUT/CHECK_HALF_ONLY.txt"
    return 0
  fi
  # Never delete the model that is serving, or the one we would roll back to. Scoring it
  # returns exactly 0.640, which is not "above the bar", and the cleanup below once ate it:
  # the rollback after allbg's bad validation failed because of that.
  case "$run/$e" in
    synth400_11s_neighbours_0919-1604/epoch15) return 0 ;;
  esac
  # Keep the disk in check: the .pt and the export are ~40 MB each. Still on IDUN if wanted again.
  rm -rf "runs/$run/weights/${e}_int8_openvino_model" "runs/$run/weights/$e.pt"
}

runs_list() {
  ssh -o BatchMode=yes idun "cd ~/nordic-cup/drone-flyby/runs && ls -d $RUNS_GLOB 2>/dev/null"
}

for round in $(seq 400); do
  # IDUN is only reachable on eduroam or the NTNU VPN. Off it, every ssh times out;
  # wait for the network to come back instead of treating it as "everything is done".
  if ! ssh -o BatchMode=yes -o ConnectTimeout=15 idun 'echo ok' >/dev/null 2>&1; then
    echo "$(date +%H:%M) IDUN unreachable (VPN down?), waiting"
    sleep 180
    continue
  fi
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
  if [ "$(wc -l < "$TSV")" -ge 40 ]; then
    q=$(ssh -o BatchMode=yes -o ConnectTimeout=15 idun 'squeue -u $USER -h' 2>/dev/null) || q=UNREACHABLE
    [ -n "$q" ] || break   # reachable and nothing queued: done. Unreachable keeps us looping.
  fi
done
echo "=== finished $(date -Is) ==="
echo finished >> "$TSV"
