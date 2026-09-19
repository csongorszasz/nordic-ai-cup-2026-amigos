"""Reproducible oracle, detector, and HTTP closed-loop comparisons."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DroneFlybyConfig
from core import build_pipeline
from core.detector import create_detector
from dtos import DroneFlybyPredictResponseDto
from local_evaluator import fetch_server_stats, frame_numbers, replay, score, wait_for_endpoint
from offline.camera_simulator import run_simulation
from offline.record_dataset import load_recorded_request
from offline.summarize_recording import frame_coverage
from utils import validate_response


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def checkpoint_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run_http(config: DroneFlybyConfig, scene: str, output: Path, simulate_latency_ms: float = 0.0,
             capture_inputs: bool = False) -> dict:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ)
    run_nonce = uuid.uuid4().hex
    environment.update({
        "PYTHONPATH": str(SRC_ROOT),
        "DRONE_FLYBY_DEVICE": config.DEVICE,
        "DRONE_FLYBY_DETECTOR_TYPE": config.DETECTOR_TYPE,
        "DRONE_FLYBY_YOLO_WEIGHTS_PATH": str(config.YOLO_WEIGHTS_PATH),
        "DRONE_FLYBY_YOLO_AUX_WEIGHTS_PATH": str(config.YOLO_AUX_WEIGHTS_PATH or ""),
        "DRONE_FLYBY_POLICY_TYPE": config.POLICY_TYPE,
        "DRONE_FLYBY_TRACKER_TYPE": config.TRACKER_TYPE,
        "DRONE_FLYBY_RECORD_VALIDATION_DATA": "1" if capture_inputs else "0",
        "DRONE_FLYBY_INFERENCE_IMAGE_SIZE": str(config.INFERENCE_IMAGE_SIZE),
        "DRONE_FLYBY_INFERENCE_IMAGE_SIZES": (
            ",".join(str(size) for size in config.INFERENCE_IMAGE_SIZES)
            if config.INFERENCE_IMAGE_SIZES is not None else ""
        ),
        "DRONE_FLYBY_INFERENCE_RECT": str(config.INFERENCE_RECT),
        "DRONE_FLYBY_INFERENCE_HALF": str(config.INFERENCE_HALF),
        "DRONE_FLYBY_DETECTOR_NMS_IOU": str(config.DETECTOR_NMS_IOU),
        "DRONE_FLYBY_DETECTOR_MAX_DET": str(config.DETECTOR_MAX_DET),
        "DRONE_FLYBY_RUN_NONCE": run_nonce,
    })
    if capture_inputs:
        environment["DRONE_FLYBY_RECORD_DIR"] = str((output / "recorded_validation_data").resolve())
        environment["DRONE_FLYBY_CAPTURE_PROVENANCE"] = json.dumps({
            "checkpoint_sha256": checkpoint_hash(config.YOLO_WEIGHTS_PATH),
            "config": asdict(config), "run_nonce": run_nonce,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "snapshot_sha256": checkpoint_hash(Path("snapshot.json")) if Path("snapshot.json").exists() else None,
        }, default=str)
    url = f"http://127.0.0.1:{port}/predict"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            cwd=SRC_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            if not wait_for_endpoint(url) or process.poll() is not None:
                raise RuntimeError(f"Benchmark endpoint failed; inspect {output / 'server.log'}")
            server = fetch_server_stats(url)
            if server is None or server.get("run_nonce") != run_nonce:
                raise RuntimeError("Benchmark port is not serving the process launched for this run")
            predictions, statistics = replay(url, scene, True, simulate_latency_ms, False)
            map50, per_class = score(scene, predictions)
            return {
                "map50": map50, "per_class": per_class,
                "scene": scene, "evaluation_frames": frame_numbers(scene),
                "simulate_latency_ms": simulate_latency_ms,
                "predictions": predictions, "statistics": asdict(statistics),
                "server": fetch_server_stats(url),
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def compare_recorded_response(original: dict, actual: dict, width: int, height: int) -> dict:
    if width <= 0 or height <= 0:
        raise ValueError("Response comparison requires positive source dimensions")
    old, new = original["annotations"], actual["annotations"]
    aligned = (
        original["request_id"] == actual["request_id"] and original["frame"] == actual["frame"]
        and original["requested_view"] == actual["requested_view"] and len(old) == len(new)
        and all(a["object_id"] == b["object_id"] for a, b in zip(old, new))
    )
    return {
        "exact": original == actual,
        "identity_classes_order_camera_aligned": aligned,
        "max_bbox_delta_source_pixels": max((
            abs(x - y) * scale for a, b in zip(old, new)
            for x, y, scale in zip(a["bbox"], b["bbox"], (width, height, width, height))
        ), default=0.0) if aligned else None,
        "max_confidence_delta": max((
            abs(a["confidence"] - b["confidence"]) for a, b in zip(old, new)
        ), default=0.0) if aligned else None,
    }


def run_recorded_views(config: DroneFlybyConfig, recording: Path,
                       expected_frames: int | None = None) -> dict:
    """Compare inference on exact fixed observations, not a counterfactual flight."""
    recording = recording.resolve()
    role = json.loads((recording / "data_role.json").read_text())
    if role.get("data_role") != "evaluation-only":
        raise ValueError("Recorded-view replay requires an evaluation-only capture")
    status = json.loads((recording / "capture_status.json").read_text())
    if any(status.get(key, 0) for key in ("dropped", "write_errors", "capture_errors", "pending_jobs")):
        raise ValueError("Recording has incomplete producer writes; inspect capture_status.json")
    rows = []
    for path in sorted((recording / "metadata").glob("frame_*.json")):
        metadata = json.loads(path.read_text())
        request = load_recorded_request(path)
        diagnostic = json.loads((recording / "diagnostics" / path.name).read_text())
        original_pipeline = diagnostic.get("pipeline")
        if not isinstance(original_pipeline, dict):
            raise ValueError("Recording lacks original pipeline diagnostics")
        if "raw_detections" not in original_pipeline:
            raise ValueError("Recording lacks raw detector diagnostics")
        order = original_pipeline.get("processing_order")
        if type(order) is not int or order < 1:
            raise ValueError("Recording lacks a verified processing order; cannot reproduce pipeline state")
        identity = (request.request_id, request.frame, request.frame_index)
        if any((entry.get("request_id"), entry.get("frame"), entry.get("frame_index")) != identity
               for entry in (diagnostic, original_pipeline)):
            raise ValueError("Recorded diagnostic identity does not match the input")
        response_path = recording / "responses" / path.name
        original = json.loads(response_path.read_text()) if response_path.exists() else None
        if original is not None and (original.get("request_id"), original.get("frame")) != identity[:2]:
            raise ValueError("Recorded response identity does not match the input")
        if original is not None:
            DroneFlybyPredictResponseDto.model_validate(original)
        nonce = metadata.get("provenance", {}).get("run_nonce")
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("Recorded replay requires serving run provenance")
        rows.append((order, nonce, path.stem, request, original, original_pipeline))
    if not rows or len(rows) != status.get("frames_received"):
        raise ValueError("Capture metadata does not account for all recorded receipts")
    if len({row[0] for row in rows}) != len(rows) or len({row[1] for row in rows}) != 1:
        raise ValueError("Recording mixes processing orders or serving instances")
    if len({row[3].sequence_id for row in rows}) != 1:
        raise ValueError("Select one recorded sequence at a time")
    rows.sort(key=lambda row: row[0])
    coverage = frame_coverage([row[3].frame_index for row in rows], expected_frames)
    pipeline = build_pipeline(config)
    pipeline.warmup()
    replayed, comparisons, numeric_comparisons = [], [], []
    raw_matches = 0
    for order, _, stem, request, original, original_pipeline in rows:
        diagnostics = {}
        response = pipeline.handle_request(request, diagnostics=diagnostics)
        validate_response(response)
        if any(event in diagnostics["events"] for event in
               ("detector_error", "pipeline_error", "memory_error", "hard_deadline_exceeded")):
            raise RuntimeError(f"Recorded-view inference failed on {request.request_id}: {diagnostics['events']}")
        payload = response.model_dump(mode="json")
        matches = payload == original if original is not None else None
        if matches is not None:
            comparisons.append(matches)
        comparison = compare_recorded_response(
            original, payload, request.original_width, request.original_height,
        ) if original is not None else None
        if comparison is not None and comparison["identity_classes_order_camera_aligned"]:
            numeric_comparisons.append(comparison)
        # Persisted diagnostics use JSON arrays for tuple-valued boxes.
        raw_json = json.loads(json.dumps(diagnostics["raw_detections"], allow_nan=False))
        raw_equal = raw_json == original_pipeline["raw_detections"]
        raw_matches += raw_equal
        replayed.append({
            "capture_stem": stem, "original_processing_order": order,
            "frame": request.frame, "frame_index": request.frame_index,
            "request_id": request.request_id, "response": payload,
            "matches_recorded_response": matches, "diagnostics": diagnostics,
            "response_comparison": comparison, "raw_detections_equal": raw_equal,
        })
    return {
        "scope": "recorded-fixed-views; not closed-loop camera or full-source evaluation",
        "recording": str(recording), "score_available": False,
        "request_count": len(rows), "coverage": coverage,
        "compared_recorded_responses": len(comparisons),
        "matching_recorded_responses": sum(comparisons),
        "all_recorded_responses_match": all(comparisons) if comparisons else None,
        "aligned_recorded_responses": len(numeric_comparisons),
        "exact_raw_detection_matches": raw_matches,
        "max_aligned_bbox_delta_source_pixels": max(
            (row["max_bbox_delta_source_pixels"] for row in numeric_comparisons), default=None,
        ),
        "max_aligned_confidence_delta": max(
            (row["max_confidence_delta"] for row in numeric_comparisons), default=None,
        ),
        "replayed_requests": replayed,
    }


def run_matrix(arguments: argparse.Namespace) -> list[dict]:
    if arguments.mode != "oracle" and arguments.weights is None:
        raise ValueError("--weights is required for real-detector experiments")
    delay = getattr(arguments, "simulate_latency_ms", 0.0)
    if not math.isfinite(delay) or delay < 0 or (delay and arguments.mode != "http"):
        raise ValueError("Simulated latency must be finite, nonnegative, and used only with HTTP replay")
    capture_inputs = getattr(arguments, "capture_inputs", False)
    if capture_inputs and arguments.mode != "http":
        raise ValueError("Request capture requires HTTP mode")
    recording = getattr(arguments, "recording", None)
    if arguments.mode == "recording":
        if recording is None or getattr(arguments, "scene", None) is not None or arguments.policies != ["hold"]:
            raise ValueError("Recorded fixed-view replay requires --recording, no --scene, and --policies hold")
    elif recording is not None:
        raise ValueError("--recording requires --mode recording")
    arguments.output.mkdir(parents=True, exist_ok=False)
    if arguments.mode != "recording":
        arguments.scene = arguments.scene or "helsinki"
    records = []
    config = DroneFlybyConfig.from_env()
    if arguments.weights is not None:
        config.YOLO_WEIGHTS_PATH = arguments.weights.resolve()
    if getattr(arguments, "aux_weights", None) is not None:
        if arguments.mode == "oracle":
            raise ValueError("An oracle comparison does not use an auxiliary detector")
        config.YOLO_AUX_WEIGHTS_PATH = arguments.aux_weights.resolve()
        config.DETECTOR_TYPE = "yolo_pair"
    if getattr(arguments, "imgsz", None) is not None:
        config.INFERENCE_IMAGE_SIZE = arguments.imgsz
        config.INFERENCE_IMAGE_SIZES = None
    if getattr(arguments, "image_sizes", None) is not None:
        config.INFERENCE_IMAGE_SIZES = tuple(arguments.image_sizes)
    write_json(arguments.output / "manifest.json", {
        "arguments": vars(arguments), "config": asdict(config),
        "config_scope": "Base configuration before tracker/policy overrides; each result stores its effective config.",
        "checkpoint_sha256": checkpoint_hash(arguments.weights),
        "aux_checkpoint_sha256": checkpoint_hash(config.YOLO_AUX_WEIGHTS_PATH)
        if config.DETECTOR_TYPE == "yolo_pair" else None,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "snapshot": json.loads(Path("snapshot.json").read_text()) if Path("snapshot.json").exists() else None,
        "status": "running",
    })
    detector = create_detector(config) if arguments.mode == "detector" else None
    if detector is not None:
        detector.warmup()
    for tracker in arguments.trackers:
        for policy in arguments.policies:
            config.TRACKER_TYPE = tracker
            config.POLICY_TYPE = policy
            output = arguments.output / f"{tracker}_{policy}"
            output.mkdir()
            started = time.monotonic()
            try:
                if arguments.mode == "http":
                    result = run_http(config, arguments.scene, output, delay, capture_inputs=capture_inputs)
                elif arguments.mode == "recording":
                    result = run_recorded_views(config, recording, getattr(arguments, "expected_frames", None))
                else:
                    result = asdict(run_simulation(
                        config, arguments.scene, detector=detector,
                        min_view_pixels=arguments.min_view_pixels,
                    ))
                result.update({
                    "status": "completed", "policy": policy, "tracker": tracker,
                    "mode": arguments.mode, "elapsed_seconds": time.monotonic() - started,
                    "config": asdict(config),
                })
                write_json(output / "result.json", result)
                summary_keys = ("policy", "tracker", "mode", "elapsed_seconds")
                summary_keys += (
                    ("scope", "score_available", "request_count", "compared_recorded_responses",
                     "matching_recorded_responses", "aligned_recorded_responses",
                     "exact_raw_detection_matches", "max_aligned_bbox_delta_source_pixels",
                     "max_aligned_confidence_delta")
                    if arguments.mode == "recording" else ("map50", "per_class")
                )
                records.append({key: result[key] for key in summary_keys})
                write_json(arguments.output / "summary.json", records)
                if arguments.mode == "recording":
                    print(f"{tracker}/{policy}: {result['request_count']} fixed requests replayed; AP unavailable", flush=True)
                else:
                    print(f"{tracker}/{policy}: AP50={result['map50']:.6f}", flush=True)
            except Exception as error:
                write_json(output / "failure.json", {
                    "status": "failed", "error": repr(error),
                    "elapsed_seconds": time.monotonic() - started,
                })
                raise
    write_json(arguments.output / "completed.json", {"status": "completed", "runs": len(records)})
    return records


def rescore_result(path: Path, scene: str | None = None) -> float:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("score_available") is False:
        raise ValueError("Recorded fixed-view replay has no ground truth or AP to rescore")
    scene = scene or result.get("scene")
    if not scene:
        raise ValueError("Supply --scene for legacy results without a scene identifier")
    predictions = {int(frame): detections for frame, detections in result["predictions"].items()}
    measured, _ = score(scene, predictions, evaluation_frames=result.get("evaluation_frames"))
    if not math.isclose(measured, result["map50"], rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Persisted predictions do not reproduce the stored score: {measured} != {result['map50']}")
    return measured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--mode", choices=["oracle", "detector", "http", "recording"])
    action.add_argument("--rescore", type=Path, help="Verify stored predictions against the common scorer.")
    parser.add_argument("--scene")
    parser.add_argument("--recording", type=Path, help="Exact evaluation-only capture; requires --mode recording.")
    parser.add_argument("--expected-frames", type=int, help="Explicit source denominator for recorded input coverage.")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--aux-weights", type=Path, help="Opt into the two-checkpoint fusion backend.")
    parser.add_argument("--imgsz", type=int, help="Explicit longest-side inference size.")
    parser.add_argument("--image-sizes", type=int, nargs=3, metavar=("L0", "L1", "L2"),
                        help="Opt-in per-zoom inference shapes; overrides scalar --imgsz.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-view-pixels", type=int, default=0)
    parser.add_argument("--simulate-latency-ms", type=float, default=0.0,
                        help="Additional HTTP request-cycle delay for cadence stress tests.")
    parser.add_argument("--capture-inputs", action="store_true",
                        help="HTTP only: save evaluation-only inputs and request-bound diagnostics.")
    parser.add_argument("--policies", nargs="+", default=["hold", "deterministic_l1", "active_coverage", "belief_voi"])
    parser.add_argument("--trackers", nargs="+", default=["passthrough", "world_map"])
    arguments = parser.parse_args()
    if arguments.rescore is not None:
        print(f"Reproduced AP50: {rescore_result(arguments.rescore, arguments.scene):.9f}")
    else:
        if arguments.output is None:
            parser.error("--output is required with --mode")
        run_matrix(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
