"""Summarize a recorded validation attempt.

    python src/offline/summarize_recording.py                       # newest sequence
    python src/offline/summarize_recording.py --dir recorded_validation_data/<seq>

Reports what the validator sent and what we answered: frame counts, annotations
per frame, class distribution, camera levels and refused commands, and latency.
There is no ground truth here, so no mAP; the official score comes from the
submission website.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import PROJECT_ROOT  # noqa: E402


def _percentile(ordered: List[float], percentile: float) -> float:
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, int(round(percentile / 100.0 * len(ordered))) - 1))
    return ordered[rank]


def _latest_session(root: Path) -> Optional[Path]:
    if not root.is_dir():
        return None
    sessions = [path for path in root.iterdir() if path.is_dir()]
    if not sessions:
        return None
    return max(sessions, key=lambda path: path.stat().st_mtime)


def summarize(session_dir: Path) -> None:
    images = sorted((session_dir / "images").glob("frame_*.png")) if (session_dir / "images").is_dir() else []
    responses = sorted((session_dir / "responses").glob("frame_*.json")) if (session_dir / "responses").is_dir() else []
    metadata = sorted((session_dir / "metadata").glob("frame_*.json")) if (session_dir / "metadata").is_dir() else []

    print(f"Sequence: {session_dir}")
    print(f"  inputs (images)   {len(images)}")
    print(f"  responses         {len(responses)}")
    print(f"  metadata          {len(metadata)}")

    annotation_counts: List[int] = []
    class_counts: Counter = Counter()
    latencies: List[float] = []
    requested_moves = 0
    for path in responses:
        payload = json.loads(path.read_text())
        annotations = payload.get("annotations", [])
        annotation_counts.append(len(annotations))
        for annotation in annotations:
            class_counts[annotation.get("object_id", "?")] += 1
        if payload.get("requested_view") is not None:
            requested_moves += 1

    if annotation_counts:
        print(
            f"  annotations/frame mean {statistics.mean(annotation_counts):.1f} "
            f"/ min {min(annotation_counts)} / max {max(annotation_counts)}"
        )
        print(f"  empty frames      {sum(1 for n in annotation_counts if n == 0)}")

    levels: Counter = Counter()
    refused = 0
    reasons: Counter = Counter()
    for path in metadata:
        entry = json.loads(path.read_text())
        levels[entry.get("resolution_level", -1)] += 1
        feedback = entry.get("camera_command_feedback")
        if feedback:
            refused += 1
            reasons[str(feedback.get("reason", "unknown"))[:80]] += 1
    if levels:
        distribution = ", ".join(f"L{level}:{count}" for level, count in sorted(levels.items()))
        print(f"  camera levels     {distribution}")
        print(f"  requested moves   {requested_moves}")
        print(f"  refused commands  {refused}")
        for reason, count in reasons.most_common(5):
            print(f"      {count:4d}x {reason}")

    index_path = session_dir / "index.jsonl"
    if index_path.is_file():
        for line in index_path.read_text().splitlines():
            try:
                latencies.append(float(json.loads(line).get("elapsed_ms", 0.0)))
            except (ValueError, json.JSONDecodeError):
                continue
    if latencies:
        ordered = sorted(latencies)
        print(
            f"  latency ms        mean {statistics.mean(ordered):.0f} / p50 {_percentile(ordered, 50):.0f} "
            f"/ p95 {_percentile(ordered, 95):.0f} / p99 {_percentile(ordered, 99):.0f} / max {ordered[-1]:.0f}"
        )

    if class_counts:
        print("  classes emitted:")
        for name, count in class_counts.most_common():
            print(f"    {name:16s} {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a recorded validation attempt.")
    parser.add_argument("--dir", type=Path, default=None, help="Session directory to summarize.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "recorded_validation_data",
        help="Where sequences are stored when --dir is omitted.",
    )
    arguments = parser.parse_args()

    session_dir = arguments.dir or _latest_session(arguments.root)
    if session_dir is None or not session_dir.is_dir():
        print(f"No recording found under {arguments.root}", file=sys.stderr)
        return 1

    summarize(session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
