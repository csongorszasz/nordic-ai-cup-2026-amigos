"""Local ASR with word-level timestamps and a transcript cache.

Wraps ``faster-whisper`` (CTranslate2 Whisper). The whole conversation is
transcribed once per request; every word keeps its start/end so evidence spans
can be built at phrase level later.

The model is loaded lazily. Call :func:`warm_up` at import time in the serving
path (the first inference is the slowest and there is no grace period), but not
in the test suite, which stubs transcription instead.

Environment overrides:

* ``WHISPER_MODEL``         default ``large-v3``
* ``WHISPER_DEVICE``        default ``cuda`` (falls back to CPU)
* ``WHISPER_COMPUTE_TYPE``  default ``float16`` (falls back to ``int8`` on CPU)
* ``WHISPER_LANGUAGE``      default ``en``
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")

# Transcript caching can be disabled for serving (cache correctness > speed).
CACHE_ENABLED = os.environ.get("MEDAPP_ASR_CACHE", "1") != "0"

_model = None


def _resolve_device() -> tuple:
    """Pick a device/compute_type pair that actually works here."""
    device, compute_type = DEVICE, COMPUTE_TYPE
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable; using CPU int8.")
                device, compute_type = "cpu", "int8"
        except Exception:  # pragma: no cover - torch import issues
            logger.warning("Could not import torch; using CPU int8.")
            device, compute_type = "cpu", "int8"
    return device, compute_type


def get_model():
    """Load and memoise the Whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        device, compute_type = _resolve_device()
        logger.info(
            "Loading Whisper %s (device=%s, compute_type=%s)",
            MODEL_SIZE,
            device,
            compute_type,
        )
        attempts = []
        for candidate in (compute_type, "int8", "float32"):
            if candidate not in attempts:
                attempts.append(candidate)
        last_error: Optional[Exception] = None
        for candidate in attempts:
            try:
                _model = WhisperModel(MODEL_SIZE, device=device, compute_type=candidate)
                if candidate != compute_type:
                    logger.warning(
                        "compute_type=%s unavailable here; using %s.",
                        compute_type,
                        candidate,
                    )
                break
            except Exception as exc:
                last_error = exc
                logger.warning("compute_type=%s failed (%s).", candidate, exc)
        if _model is None:
            raise last_error if last_error else RuntimeError("ASR load failed")
    return _model


def warm_up() -> None:
    """Load the model and run one tiny inference to trigger CUDA kernels.

    The first real request is the slowest and there is no warm-up budget, so
    this is called at import time in the server.
    """
    model = get_model()
    try:
        import numpy as np

        silence = np.zeros(16000, dtype="float32")
        list(model.transcribe(silence, language=LANGUAGE)[0])
    except Exception:  # pragma: no cover - warm-up is best-effort
        logger.exception("Whisper warm-up inference failed (continuing).")


def config_hash() -> str:
    """Short hash of the decoding configuration a transcript was made with.

    Folded into the cache filename so swapping the model or decoding options
    (e.g. to ``large-v3-turbo`` for serving) never silently reuses a transcript
    produced by different settings.
    """
    payload = json.dumps(
        {
            "model": MODEL_SIZE,
            "compute_type": COMPUTE_TYPE,
            "language": LANGUAGE,
            "vad_filter": True,
            "condition_on_previous_text": False,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def content_hash(audio_bytes: bytes) -> str:
    """Short content hash of the audio, so a repeated filename cannot alias."""
    return hashlib.sha256(audio_bytes).hexdigest()[:8]


def cache_path(audio_filename: str, audio_bytes: Optional[bytes] = None) -> Path:
    """Where the transcript for one conversation is cached.

    The key combines the filename, the decoding configuration and (when the
    bytes are supplied) the audio content, so reusing a filename for different
    audio can never serve a stale transcript.
    """
    key = f"{Path(audio_filename).stem}.{config_hash()}"
    if audio_bytes is not None:
        key += f".{content_hash(audio_bytes)}"
    return TRANSCRIPTS_DIR / f"{key}.json"


def transcribe_bytes(
    audio_bytes: bytes,
    audio_filename: str,
    cache: bool = True,
    force: bool = False,
) -> Dict:
    """Transcribe one conversation, returning a transcript dict.

    The returned dict carries both segments and a flat ``words`` list (each word
    tagged with the ``seg_idx`` it came from), so downstream code can build
    phrase spans without re-running ASR. Results are cached unless
    ``MEDAPP_ASR_CACHE=0`` or ``cache=False``.
    """
    cache = cache and CACHE_ENABLED
    path = cache_path(audio_filename, audio_bytes)
    if cache and not force and path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.warning("Corrupt transcript cache %s; re-transcribing.", path)

    model = get_model()

    with tempfile.NamedTemporaryFile(suffix=".mp3") as handle:
        handle.write(audio_bytes)
        handle.flush()
        segment_iter, info = model.transcribe(
            handle.name,
            language=LANGUAGE,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        segments: List[Dict] = []
        words: List[Dict] = []
        for segment in segment_iter:
            seg_idx = len(segments)
            segment_words = []
            for word in segment.words or []:
                entry = {
                    "start": float(word.start),
                    "end": float(word.end),
                    "word": word.word,
                    "probability": float(word.probability),
                    "seg_idx": seg_idx,
                }
                words.append(entry)
                segment_words.append(entry)
            segments.append(
                {
                    "id": seg_idx,
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "avg_logprob": float(segment.avg_logprob),
                    "no_speech_prob": float(segment.no_speech_prob),
                    "compression_ratio": float(segment.compression_ratio),
                    "temperature": (
                        float(segment.temperature)
                        if segment.temperature is not None
                        else None
                    ),
                }
            )

    transcript = {
        "audio_filename": audio_filename,
        "model": MODEL_SIZE,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "duration_after_vad": float(getattr(info, "duration_after_vad", 0.0)),
        "segments": segments,
        "words": words,
    }

    if cache:
        TRANSCRIPTS_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(transcript, indent=2))
        logger.info("Cached transcript -> %s", path)

    return transcript
