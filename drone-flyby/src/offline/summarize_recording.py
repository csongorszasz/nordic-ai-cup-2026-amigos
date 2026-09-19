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
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import PROJECT_ROOT  # noqa: E402
from offline.record_dataset import load_recorded_request


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


def frame_coverage(indices: List[int], expected_frames: Optional[int] = None) -> dict:
    if expected_frames is not None and expected_frames <= 0:
        raise ValueError("Expected frame count must be positive")
    counts = Counter(indices)
    observed = sorted(counts)
    upper = expected_frames if expected_frames is not None else (max(observed) + 1 if observed else 0)
    missing = []
    cursor = 0
    for index in observed:
        if index >= upper:
            continue
        if index > cursor:
            missing.append([cursor, index])
        cursor = index + 1
    if cursor < upper:
        missing.append([cursor, upper])
    return {
        "expected_frames": expected_frames, "unique_indices": len(observed),
        "received_frame_indices": observed,
        "duplicate_indices": {str(index): count for index, count in counts.items() if count > 1},
        "missing_index_ranges_half_open": missing,
        "missing_count": sum(end - start for start, end in missing),
        "tail_coverage_known": expected_frames is not None,
        "out_of_expected_range": [index for index in observed if index >= upper],
    }


def summarize(session_dir: Path, expected_frames: Optional[int] = None) -> dict:
    images = sorted((session_dir / "images").glob("frame_*.png")) if (session_dir / "images").is_dir() else []
    responses = sorted((session_dir / "responses").glob("frame_*.json")) if (session_dir / "responses").is_dir() else []
    metadata = sorted((session_dir / "metadata").glob("frame_*.json")) if (session_dir / "metadata").is_dir() else []
    diagnostics = sorted((session_dir / "diagnostics").glob("frame_*.json"))
    integrity_errors = []
    image_stems, response_stems = {path.stem for path in images}, {path.stem for path in responses}
    metadata_stems = {path.stem for path in metadata}
    diagnostic_stems = {path.stem for path in diagnostics}
    indices, replayable, unknown_indices = [], 0, 0
    response_identities, metadata_identities = {}, {}

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
        response_identities[path.stem] = (payload.get("request_id"), payload.get("frame"))
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
        index = entry.get("frame_index")
        if type(index) is int and index >= 0:
            indices.append(index)
        else:
            unknown_indices += 1
        if entry.get("schema_version") == 2:
            metadata_identities[path.stem] = (entry["request_id"], entry["frame"], entry["frame_index"])
            if (path.stem in response_identities
                    and response_identities[path.stem] != metadata_identities[path.stem][:2]):
                integrity_errors.append(f"{path.name}: response identity mismatch")
            try:
                load_recorded_request(path)
                replayable += 1
            except (OSError, ValueError, KeyError) as error:
                integrity_errors.append(f"{path.name}: {error}")
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
        for line_number, line in enumerate(index_path.read_text().splitlines(), 1):
            try:
                value = float(json.loads(line)["elapsed_ms"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Latency must be finite and nonnegative")
                latencies.append(value)
            except (ValueError, KeyError, TypeError) as error:
                integrity_errors.append(f"index.jsonl:{line_number}: {error}")
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

    raw_classes, events, output_sources = Counter(), Counter(), Counter()
    raw_confidences = []
    detector_observations = 0
    pipeline_diagnostic_records = 0
    for path in diagnostics:
        entry = json.loads(path.read_text())
        expected_identity = metadata_identities.get(path.stem)
        if expected_identity is not None and (
                entry.get("request_id"), entry.get("frame"), entry.get("frame_index")) != expected_identity:
            integrity_errors.append(f"{path.name}: diagnostic identity mismatch")
        pipeline = entry.get("pipeline")
        if pipeline is None:
            continue
        pipeline_diagnostic_records += 1
        if expected_identity is not None and (
                pipeline.get("request_id"), pipeline.get("frame"), pipeline.get("frame_index")) != expected_identity:
            integrity_errors.append(f"{path.name}: pipeline identity mismatch")
        events.update(pipeline["events"])
        output_sources.update(pipeline.get("output_sources", []))
        raw = pipeline.get("raw_detections")
        if raw is not None:
            detector_observations += 1
            for detection in raw:
                confidence = float(detection["confidence"])
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    integrity_errors.append(f"{path.name}: invalid raw confidence")
                    continue
                raw_classes[detection["class_name"]] += 1
                raw_confidences.append(confidence)

    status_path = session_dir / "capture_status.json"
    capture_status = json.loads(status_path.read_text()) if status_path.exists() else None
    coverage = frame_coverage(indices, expected_frames)
    missing_pairs = {
        "images": sorted(metadata_stems - image_stems),
        "responses": sorted(metadata_stems - response_stems),
        "diagnostics": sorted(metadata_stems - diagnostic_stems),
        "metadata_for_images": sorted(image_stems - metadata_stems),
        "metadata_for_responses": sorted(response_stems - metadata_stems),
        "metadata_for_diagnostics": sorted(diagnostic_stems - metadata_stems),
    }
    failed_capture = capture_status is None or any(
        capture_status.get(key, 0) for key in ("dropped", "write_errors", "capture_errors", "pending_jobs")
    )
    complete = (
        expected_frames is not None and coverage["missing_count"] == 0
        and not coverage["out_of_expected_range"] and not unknown_indices
        and replayable == len(metadata) and not integrity_errors
        and not any(missing_pairs.values()) and not failed_capture
    )
    report = {
        "score_available": False, "evaluator_acceptance_known": False,
        "complete_capture": complete, "coverage": coverage,
        "images": len(images), "responses": len(responses), "metadata": len(metadata),
        "replayable_requests": replayable, "unknown_frame_index_records": unknown_indices,
        "missing_artifacts": missing_pairs, "integrity_errors": integrity_errors,
        "capture_status": capture_status, "classes_emitted": dict(class_counts),
        "raw_classes": dict(raw_classes), "detector_observations": detector_observations,
        "pipeline_diagnostic_records": pipeline_diagnostic_records,
        "complete_pipeline_diagnostics": complete and pipeline_diagnostic_records == len(metadata),
        "raw_confidence_median": statistics.median(raw_confidences) if raw_confidences else None,
        "raw_below_0_01": sum(value < 0.01 for value in raw_confidences),
        "pipeline_events": dict(events), "output_sources": dict(output_sources),
    }
    print(f"  unique frame indices {coverage['unique_indices']}")
    print(f"  missing indices      {coverage['missing_count']} (tail known: {coverage['tail_coverage_known']})")
    print(f"  complete capture     {complete} (not evaluator acceptance)")
    if integrity_errors:
        print("  recording integrity errors:")
        for error in integrity_errors:
            print(f"    {error}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a recorded validation attempt.")
    parser.add_argument("--dir", type=Path, default=None, help="Session directory to summarize.")
    parser.add_argument("--expected-frames", type=int, help="Explicit attempt denominator; validation has 249 frames.")
    parser.add_argument("--output-json", type=Path, help="Persist counts and integrity findings, never fabricated AP.")
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

    report = summarize(session_dir, arguments.expected_frames)
    if arguments.output_json is not None:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
