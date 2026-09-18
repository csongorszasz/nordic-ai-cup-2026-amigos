"""Record validator inputs and server responses during a validation attempt.

The evaluator drives the server from outside; this captures everything so the
sequence can be inspected afterwards. All disk I/O happens on a background
daemon thread, so the request path only enqueues a small job and never blocks on
a PNG or JSON write.

Per sequence, under ``<output_dir>/<sequence_id>/``:

* ``images/frame_NNNNNN.png``    the exact PNG the validator sent (no re-encode)
* ``metadata/frame_NNNNNN.json`` request metadata (camera, region, feedback)
* ``responses/frame_NNNNNN.json`` our annotations, requested view and latency
* ``index.jsonl``                one summary line per frame

Enable with ``DRONE_FLYBY_RECORD_VALIDATION_DATA=1`` and optionally
``DRONE_FLYBY_RECORD_DIR=<path>``.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


class ValidationDatasetRecorder:
    """Asynchronously saves incoming frames and outgoing responses to disk."""

    def __init__(
        self,
        output_dir: Path = PROJECT_ROOT / "recorded_validation_data",
        max_queue: int = 1024,
    ):
        self.output_dir = Path(output_dir)
        self.max_queue = max_queue
        self.enabled = False
        self.session_dir: Optional[Path] = None

        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=max_queue)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.dropped = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self, sequence_id: str) -> None:
        """Initialize the directory structure for a recording session."""
        session_dir = self.output_dir / sequence_id
        with self._lock:
            self.session_dir = session_dir
            for name in ("images", "metadata", "responses"):
                (session_dir / name).mkdir(parents=True, exist_ok=True)
            self.enabled = True
        self._ensure_worker()
        logger.info("Recording validation sequence to %s", session_dir)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._run, name="validation-recorder", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        self.enabled = False

    def flush(self, timeout: float = 15.0) -> None:
        """Block until the queue has drained (bounded by ``timeout``)."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)

    def shutdown(self) -> None:
        self.flush()
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    # Enqueue from the request path (non-blocking)
    # ------------------------------------------------------------------ #

    def _enqueue(self, job: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 100 == 1:
                logger.warning(
                    "Validation recorder queue full; dropped %d jobs", self.dropped
                )

    def record_frame(self, request: DroneFlybyPredictRequestDto) -> None:
        """Queue one incoming validator frame."""
        if not self.enabled or self.session_dir is None:
            return

        feedback = request.camera_command_feedback
        job = {
            "kind": "frame",
            "session_dir": self.session_dir,
            "frame": request.frame,
            "frame_index": request.frame_index,
            "request_id": request.request_id,
            "image_b64": request.view.image,
            "meta": {
                "sequence_id": request.sequence_id,
                "frame": request.frame,
                "frame_index": request.frame_index,
                "request_id": request.request_id,
                "resolution_level": request.view.resolution_level,
                "center_x": request.view.center_x,
                "center_y": request.view.center_y,
                "source_region_xyxy": list(request.view.source_region_xyxy),
                "frame_interval_ms": request.frame_interval_ms,
                "response_timeout_ms": request.response_timeout_ms,
                "camera_command_feedback": _jsonable(feedback),
            },
        }
        self._enqueue(job)

    def record_response(
        self,
        request: DroneFlybyPredictRequestDto,
        response: DroneFlybyPredictResponseDto,
        elapsed_ms: float,
    ) -> None:
        """Queue the response we returned for one request."""
        if not self.enabled or self.session_dir is None:
            return

        payload = _jsonable(response)
        annotations = payload.get("annotations", []) if isinstance(payload, dict) else []
        requested_view = payload.get("requested_view") if isinstance(payload, dict) else None
        job = {
            "kind": "response",
            "session_dir": self.session_dir,
            "frame": request.frame,
            "frame_index": request.frame_index,
            "request_id": request.request_id,
            "elapsed_ms": round(float(elapsed_ms), 3),
            "response": payload,
            "summary": {
                "frame": request.frame,
                "frame_index": request.frame_index,
                "request_id": request.request_id,
                "num_annotations": len(annotations),
                "requested_view": requested_view,
                "elapsed_ms": round(float(elapsed_ms), 3),
            },
        }
        self._enqueue(job)

    # ------------------------------------------------------------------ #
    # Background writer
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if job["kind"] == "frame":
                    self._write_frame(job)
                else:
                    self._write_response(job)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to record a validation artifact")
            finally:
                self._queue.task_done()

    @staticmethod
    def _write_frame(job: Dict[str, Any]) -> None:
        session_dir: Path = job["session_dir"]
        frame_index = job["frame_index"]
        # Persist the received PNG bytes verbatim: no decode/re-encode.
        (session_dir / "images" / f"frame_{frame_index:06d}.png").write_bytes(
            base64.b64decode(job["image_b64"])
        )
        (session_dir / "metadata" / f"frame_{frame_index:06d}.json").write_text(
            json.dumps(job["meta"], indent=2)
        )

    @staticmethod
    def _write_response(job: Dict[str, Any]) -> None:
        session_dir: Path = job["session_dir"]
        frame_index = job["frame_index"]
        (session_dir / "responses" / f"frame_{frame_index:06d}.json").write_text(
            json.dumps(job["response"], indent=2)
        )
        with open(session_dir / "index.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(job["summary"]) + "\n")


def recorder_from_env() -> Optional[ValidationDatasetRecorder]:
    """Build a recorder when validation capture is enabled via environment."""
    enabled_value = os.getenv("DRONE_FLYBY_RECORD_VALIDATION_DATA", "0").strip().lower()
    if enabled_value not in {"1", "true", "yes", "on"}:
        return None

    output_dir = Path(os.getenv("DRONE_FLYBY_RECORD_DIR", str(PROJECT_ROOT / "recorded_validation_data")))
    max_queue = int(os.getenv("DRONE_FLYBY_RECORD_MAX_QUEUE", "1024"))
    recorder = ValidationDatasetRecorder(output_dir=output_dir, max_queue=max_queue)
    atexit.register(recorder.shutdown)
    return recorder
