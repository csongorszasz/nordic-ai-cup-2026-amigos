#!/usr/bin/env bash
# Download the drone-flyby data that is too big for git from the team's GitHub release
# and unpack it into place. Run from anywhere; needs `gh auth login` with repo access.
#
#   bash training/fetch_data.sh                 # model + frames + recordings
#   bash training/fetch_data.sh model frames    # pick: model | frames | recordings
#   FORCE=1 bash training/fetch_data.sh model   # overwrite existing files (e.g. your sprite edits)
#
# model      -> runs/yolo11s_baseline/weights/best.pt, sprites/            (36 MB)
# frames     -> recordings/validation_4k/, recordings/RUNS.md              (854 MB)
# recordings -> recordings/<run>/ raw evaluator views of 7 validation runs (1.4 GB)
set -euo pipefail

REPO="csongorszasz/nordic-ai-cup-2026-amigos"
TAG="drone-data-2026-09-17"
DRONE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$DRONE_DIR/.downloads"

declare -A ASSET=(
  [model]="drone-baseline-model-sprites.zip"
  [frames]="drone-validation-4k.zip"
  [recordings]="drone-validation-recordings.zip"
)

command -v gh >/dev/null || { echo "gh CLI not found: https://cli.github.com"; exit 1; }
command -v unzip >/dev/null || { echo "unzip not found (sudo apt install unzip)"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Run 'gh auth login' first"; exit 1; }

parts=("$@")
[ ${#parts[@]} -eq 0 ] && parts=(model frames recordings)

mkdir -p "$CACHE"
gh release download "$TAG" -R "$REPO" -p SHA256SUMS -D "$CACHE" --clobber

for part in "${parts[@]}"; do
  file="${ASSET[$part]:-}"
  [ -n "$file" ] || { echo "unknown part '$part' (use: model frames recordings)"; exit 1; }
  expected="$(grep " $file\$" "$CACHE/SHA256SUMS" | cut -d' ' -f1)"

  if [ -f "$CACHE/$file" ] && [ "$(sha256sum "$CACHE/$file" | cut -d' ' -f1)" = "$expected" ]; then
    echo "== $part: $file already downloaded"
  else
    echo "== $part: downloading $file"
    gh release download "$TAG" -R "$REPO" -p "$file" -D "$CACHE" --clobber
    actual="$(sha256sum "$CACHE/$file" | cut -d' ' -f1)"
    [ "$actual" = "$expected" ] || { echo "checksum mismatch for $file"; exit 1; }
  fi

  # Never overwrite local work unless FORCE=1 (e.g. sprites/manual.json edits).
  if [ "${FORCE:-0}" = "1" ]; then overwrite=-o; else overwrite=-n; fi
  echo "== $part: unpacking into $DRONE_DIR"
  unzip -q $overwrite "$CACHE/$file" -d "$DRONE_DIR"
done

echo "Done. Downloads cached in $CACHE (safe to delete)."
