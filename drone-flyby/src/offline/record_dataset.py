"""Bounded, asynchronous evaluation-only capture of complete prediction requests.

Each receipt has a unique ``frame_<index>_<capture-id>`` stem, so retries and
interleaved sequences cannot overwrite or misattribute an earlier observation.
PNG bytes, replayable DTO metadata, responses, and diagnostics share that stem.
"""

from __future__ import annotations

import atexit
import base64
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Optional
import uuid

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capture:
    sequence_id: str
    request_id: str
    frame: int
    frame_index: int
    stem: str


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_recorded_request(metadata_path: Path) -> DroneFlybyPredictRequestDto:
    """Load exact recorded input, never reinterpret a transmitted crop as 4K."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or "request" not in metadata:
        raise ValueError("Legacy recording lacks complete replayable request metadata")
    image = (metadata_path.parent.parent / "images" / f"{metadata_path.stem}.png").read_bytes()
    if hashlib.sha256(image).hexdigest() != metadata["image_sha256"]:
        raise ValueError(f"Recorded image hash mismatch: {metadata_path.stem}")
    payload = metadata["request"]
    payload["view"]["image"] = base64.b64encode(image).decode("ascii")
    request = DroneFlybyPredictRequestDto.model_validate(payload)
    for key in ("sequence_id", "request_id", "frame", "frame_index"):
        if getattr(request, key) != metadata[key]:
            raise ValueError(f"Recorded request identity mismatch: {key}")
    return request


class ValidationDatasetRecorder:
    """Capture failures are observable but must not break prediction."""

    def __init__(self, output_dir: Path = PROJECT_ROOT / "recorded_validation_data",
                 max_queue: int = 1024, provenance: Optional[dict] = None):
        if max_queue <= 0:
            raise ValueError("Recording queue must have a positive bound")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._mark_evaluation_only(self.output_dir)
        self.provenance = dict(provenance or {})
        self.max_queue = max_queue
        self.enabled = False
        self.session_dir: Optional[Path] = None
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=max_queue)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._totals = Counter()
        self._sequences: dict[str, Counter] = {}
        self._last_errors: dict[str, str] = {}
        self._dirty: set[str] = set()
        self._status_dirty = False
        self._ordinal = 0

    @staticmethod
    def _mark_evaluation_only(directory: Path) -> None:
        marker = directory / "data_role.json"
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")).get("data_role") != "evaluation-only":
                raise ValueError(f"Refusing to overwrite another data role: {directory}")
        else:
            _write_json(marker, {"data_role": "evaluation-only"})

    def _session_path(self, sequence_id: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", sequence_id) is None:
            raise ValueError("Unsafe recording sequence_id; use letters, digits, hyphens or underscores")
        path = (self.output_dir / sequence_id).resolve()
        if not path.is_relative_to(self.output_dir):
            raise ValueError("Recording sequence path escapes its owned root")
        return path

    def _prepare_directory(self, sequence_id: str) -> Path:
        path = self._session_path(sequence_id)
        path.mkdir(parents=True, exist_ok=True)
        self._mark_evaluation_only(path)
        for name in ("images", "metadata", "responses", "diagnostics"):
            child = (path / name).resolve()
            if not child.is_relative_to(path):
                raise ValueError("Recording artifact directory escapes its sequence")
            child.mkdir(exist_ok=True)
        return path

    def start(self, sequence_id: str) -> None:
        path = self._session_path(sequence_id)
        with self._lock:
            self.session_dir = path
            self._sequences.setdefault(sequence_id, Counter())
            self.enabled = True
            if self._worker is None or not self._worker.is_alive():
                self._stop.clear()
                self._worker = threading.Thread(target=self._run, name="validation-recorder", daemon=True)
                self._worker.start()

    def stop(self) -> None:
        self.enabled = False

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._totals["dropped"]

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled, "sequence_count": len(self._sequences),
                "pending_jobs": self._queue.unfinished_tasks,
                **{key: self._totals[key] for key in (
                    "frames_received", "responses_generated", "frames_written", "responses_written",
                    "diagnostics_written", "dropped", "write_errors", "capture_errors",
                )},
            }

    def _count(self, sequence_id: Optional[str], key: str) -> None:
        with self._lock:
            self._totals[key] += 1
            self._status_dirty = True
            if sequence_id is not None:
                self._sequences.setdefault(sequence_id, Counter())[key] += 1
                self._dirty.add(sequence_id)

    def report_error(self, message: str, sequence_id: Optional[str] = None) -> None:
        self._count(sequence_id, "capture_errors")
        logger.error("Validation capture failed: %s", message)

    def flush(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._queue.unfinished_tasks:
            raise TimeoutError(f"Recording did not drain {self._queue.unfinished_tasks} pending jobs")
        counts = self.stats()
        if any(counts[key] for key in ("write_errors", "capture_errors", "dropped")):
            raise RuntimeError(f"Recording is incomplete; inspect recorder_status.json: {counts}")

    def shutdown(self) -> None:
        try:
            self.flush()
        finally:
            with self._lock:
                self.enabled = False
                self._status_dirty = True
            self._stop.set()
            if self._worker is not None:
                self._worker.join(timeout=5)
                if self._worker.is_alive():
                    raise TimeoutError("Validation recorder worker did not stop")
            self._persist_statuses()

    def _enqueue(self, job: dict) -> None:
        with self._lock:
            try:
                self._queue.put_nowait(job)
                self._count(job["capture"].sequence_id, "jobs_queued")
            except queue.Full:
                self._count(job["capture"].sequence_id, "dropped")
                logger.warning("Validation recorder queue full; dropped %s for %s",
                               job["kind"], job["capture"].request_id)

    def record_frame(self, request: DroneFlybyPredictRequestDto) -> Optional[Capture]:
        if not self.enabled:
            return None
        self._session_path(request.sequence_id)
        with self._lock:
            self._ordinal += 1
            ordinal = self._ordinal
        capture = Capture(request.sequence_id, request.request_id, request.frame, request.frame_index,
                          f"frame_{request.frame_index:06d}_{uuid.uuid4().hex}")
        payload = request.model_dump(mode="json", exclude={"view": {"image"}})
        metadata = {
            "schema_version": 2, "received_at": datetime.now(timezone.utc).isoformat(),
            "received_monotonic": time.monotonic(), "receipt_ordinal": ordinal,
            "sequence_id": request.sequence_id, "request_id": request.request_id,
            "frame": request.frame, "frame_index": request.frame_index,
            "resolution_level": request.view.resolution_level,
            "center_x": request.view.center_x, "center_y": request.view.center_y,
            "source_region_xyxy": list(request.view.source_region_xyxy),
            "camera_command_feedback": payload["camera_command_feedback"],
            "request": payload, "provenance": self.provenance,
        }
        self._count(request.sequence_id, "frames_received")
        self._enqueue({"kind": "frame", "capture": capture, "image_b64": request.view.image,
                       "metadata": metadata})
        return capture

    def record_response(self, request: DroneFlybyPredictRequestDto,
                        response: Optional[DroneFlybyPredictResponseDto], elapsed_ms: float,
                        *, capture: Optional[Capture] = None, diagnostics: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        if capture is None or (capture.sequence_id, capture.request_id, capture.frame, capture.frame_index) != (
                request.sequence_id, request.request_id, request.frame, request.frame_index):
            raise ValueError("Response capture does not match its received request")
        if response is not None and (response.request_id != request.request_id or response.frame != request.frame):
            raise ValueError("Response identity does not match its received request")
        if diagnostics is not None and diagnostics.get("request_id") != request.request_id:
            raise ValueError("Diagnostics belong to another request")
        payload = response.model_dump(mode="json") if response is not None else None
        if payload is not None:
            self._count(request.sequence_id, "responses_generated")
        summary = {
            "frame": request.frame, "frame_index": request.frame_index,
            "request_id": request.request_id, "stem": capture.stem,
            "response_generated": response is not None, "evaluator_accepted": None,
            "num_annotations": len(response.annotations) if response is not None else None,
            "requested_view": payload["requested_view"] if payload is not None else None,
            "elapsed_ms": round(float(elapsed_ms), 3),
        }
        self._enqueue({"kind": "response", "capture": capture, "response": payload,
                       "diagnostics": diagnostics, "summary": summary})

    def _persist_statuses(self, completing_job: bool = False) -> None:
        with self._lock:
            if not self._status_dirty:
                return
            names = tuple(self._dirty)
            snapshots = {name: {
                **dict(self._sequences[name]), "last_write_error": self._last_errors.get(name),
                "evaluator_acceptance_known": False,
                "pending_jobs": self._sequences[name]["jobs_queued"] - self._sequences[name]["jobs_finished"],
            } for name in names}
            totals = self.stats()
            totals["pending_jobs"] = max(0, totals["pending_jobs"] - int(completing_job))
            self._dirty.clear()
            self._status_dirty = False
        for name, snapshot in snapshots.items():
            _write_json(self._prepare_directory(name) / "capture_status.json", snapshot)
        _write_json(self.output_dir / "recorder_status.json", totals)

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                try:
                    self._persist_statuses()
                except (OSError, ValueError) as error:
                    self._count(None, "write_errors")
                    logger.exception("Failed to persist recorder status: %s", error)
                continue
            capture = job["capture"]
            try:
                directory = self._prepare_directory(capture.sequence_id)
                if job["kind"] == "frame":
                    self._write_frame(directory, job)
                    self._count(capture.sequence_id, "frames_written")
                else:
                    self._write_response(directory, job)
                    if job["response"] is not None:
                        self._count(capture.sequence_id, "responses_written")
                    self._count(capture.sequence_id, "diagnostics_written")
            except (OSError, ValueError, TypeError) as error:
                self._count(capture.sequence_id, "write_errors")
                with self._lock:
                    self._last_errors[capture.sequence_id] = str(error)[:512]
                logger.exception("Failed to record validation request %s", capture.request_id)
            finally:
                self._count(capture.sequence_id, "jobs_finished")
                try:
                    self._persist_statuses(completing_job=True)
                except (OSError, ValueError) as error:
                    self._count(None, "write_errors")
                    logger.exception("Failed to persist recorder status: %s", error)
                self._queue.task_done()

    @staticmethod
    def _write_frame(directory: Path, job: dict) -> None:
        stem = job["capture"].stem
        image = base64.b64decode(job["image_b64"], validate=True)
        image_path = directory / "images" / f"{stem}.png"
        temporary = image_path.with_suffix(".tmp")
        temporary.write_bytes(image)
        temporary.replace(image_path)
        metadata = {**job["metadata"], "image_sha256": hashlib.sha256(image).hexdigest()}
        _write_json(directory / "metadata" / f"{stem}.json", metadata)

    @staticmethod
    def _write_response(directory: Path, job: dict) -> None:
        stem = job["capture"].stem
        if job["response"] is not None:
            _write_json(directory / "responses" / f"{stem}.json", job["response"])
        _write_json(directory / "diagnostics" / f"{stem}.json", {
            **job["summary"], "pipeline": job["diagnostics"],
        })
        with (directory / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(job["summary"]) + "\n")


def recorder_from_env() -> Optional[ValidationDatasetRecorder]:
    enabled = os.getenv("DRONE_FLYBY_RECORD_VALIDATION_DATA", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    output_dir = Path(os.getenv("DRONE_FLYBY_RECORD_DIR", str(PROJECT_ROOT / "recorded_validation_data")))
    provenance = json.loads(os.getenv("DRONE_FLYBY_CAPTURE_PROVENANCE", "{}"))
    if not isinstance(provenance, dict):
        raise ValueError("Capture provenance must be a JSON object")
    recorder = ValidationDatasetRecorder(
        output_dir, max_queue=int(os.getenv("DRONE_FLYBY_RECORD_MAX_QUEUE", "1024")),
        provenance=provenance,
    )
    atexit.register(recorder.shutdown)
    return recorder
