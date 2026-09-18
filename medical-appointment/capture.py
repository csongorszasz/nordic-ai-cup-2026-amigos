"""Opt-in capture of incoming requests and our responses.

Enabled with ``MEDAPP_CAPTURE=1``. Used to record what the validation service
sends (questions, and audio with ``MEDAPP_CAPTURE_AUDIO=1``) and what we
answered, so a local run can be replayed and compared against the service's
score. Debugging only — never train on captured validation data. Audio is off
by default so an evaluation attempt does not leave the evaluation set on disk.
"""

import base64
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MEDAPP_CAPTURE", "0") == "1"
CAPTURE_AUDIO = os.environ.get("MEDAPP_CAPTURE_AUDIO", "0") == "1"
CAPTURE_DIR = Path(__file__).resolve().parent / "captured"
AUDIO_DIR = CAPTURE_DIR / "audio"


def maybe_capture(request, response, latency_s=None) -> None:
    """Persist one request/response pair. Never raises."""
    if not ENABLED:
        return
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(request.audio_filename).stem
        if CAPTURE_AUDIO:
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = AUDIO_DIR / f"{stem}.mp3"
            if not audio_path.exists():
                audio_path.write_bytes(base64.b64decode(request.audio_base64))

        record = {
            "audio_filename": request.audio_filename,
            "questions": list(request.questions),
            "answers": list(response.answers),
            "evidence_start": list(response.evidence_start),
            "evidence_end": list(response.evidence_end),
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (CAPTURE_DIR / f"{stem}.json").write_text(json.dumps(record, indent=2))
        logger.info("captured %s (%.1fs)", stem, latency_s or 0.0)
    except Exception:
        logger.exception("capture failed for %s", getattr(request, "audio_filename", "?"))
