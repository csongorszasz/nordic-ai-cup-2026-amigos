"""The model behind ``/predict``.

Transcribe once per conversation with local ASR (word timestamps kept), then
hand the questions to the answerer selected by ``MEDAPP_ANSWERER`` (see
``answerers/``). The default is the frozen NLI pipeline; a task-trained
cross-encoder can be A/B'd behind the same contract. Nothing here calls a cloud
API.

``predict`` never raises and never overruns: one exception or timeout would
score all ten of a conversation's questions wrong, and five timeouts in a row
end the attempt. So:

* the work runs in a worker thread and ``predict`` waits at most
  ``MEDAPP_HARD_TIMEOUT_S``; past that it answers with guesses and leaves the
  worker behind;
* one conversation uses the GPU at a time (``_GPU_LOCK``). A request that finds
  the lock still held by an abandoned worker waits only as long as its own
  budget allows, then guesses, so an overrun cannot cascade into the next
  request's time;
* every guess comes from the question-text prior (``prior.py``), not a constant.
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import asr
from answerers import get_answerer
from capture import maybe_capture
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from prior import guess
from utils import decode_audio

logger = logging.getLogger(__name__)

# Soft deadline handed to ASR and the answerer: they stop starting new work and
# return what they have once this much wall-clock has elapsed.
DEADLINE_S = float(os.environ.get("MEDAPP_DEADLINE_S", "50"))
# Hard ceiling on how long ``predict`` waits for the worker (budget is 60 s,
# measured by the caller from the POST, so keep room for transfer).
HARD_TIMEOUT_S = float(os.environ.get("MEDAPP_HARD_TIMEOUT_S", "55"))
# Raise at startup instead of serving a degraded model (the serve job sets 1).
STRICT_STARTUP = os.environ.get("MEDAPP_STRICT_STARTUP", "0") == "1"

_GPU_LOCK = threading.Lock()

# What /ready reports. ``ready`` flips to True only after a successful warm-up.
READINESS: Dict = {"ready": False, "errors": [], "warmup_s": None}

# Load the models at import time: the first inference is the slowest and there
# is no warm-up budget. Skipped in tests via MEDAPP_SKIP_WARMUP=1.
if os.environ.get("MEDAPP_SKIP_WARMUP") != "1":
    _started = time.time()
    try:
        guess(["Is the prior trained?"])
        asr.warm_up()
        answerer = get_answerer()
        warm_up = getattr(answerer, "warm_up", None)
        if warm_up is not None:
            warm_up()
        status = getattr(answerer, "status", None)
        READINESS.update(status() if status is not None else {"answerer": type(answerer).__name__})
        READINESS["asr_model"] = asr.MODEL_SIZE
        READINESS["ready"] = True
    except Exception as exc:  # pragma: no cover - startup environment issue
        READINESS["errors"].append(repr(exc))
        if STRICT_STARTUP:
            logger.exception("Model warm-up failed; refusing to serve (MEDAPP_STRICT_STARTUP=1).")
            raise
        logger.exception("Model warm-up failed; will load lazily on first request.")
    READINESS["warmup_s"] = round(time.time() - _started, 1)
    logger.info("readiness: %s", READINESS)


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation. Never raises."""
    started = time.time()
    outcome: Dict = {}

    def work() -> None:
        try:
            outcome["response"] = _predict(request, started)
        except Exception:
            logger.exception(
                "predict failed for %s; returning guesses.", request.audio_filename
            )

    worker = threading.Thread(target=work, name="predict", daemon=True)
    worker.start()
    worker.join(max(0.0, HARD_TIMEOUT_S - (time.time() - started)))

    response = outcome.get("response")
    if response is None:
        if worker.is_alive():
            logger.error(
                "%s overran %.0fs; answering with guesses (worker abandoned).",
                request.audio_filename, HARD_TIMEOUT_S,
            )
        response = _fallback(request.questions)

    maybe_capture(request, response, time.time() - started)
    return response


def _predict(request: ASRQuestionRequestDto, started: float) -> ASRQuestionResponseDto:
    deadline = started + DEADLINE_S
    audio_bytes = decode_audio(request.audio_base64)

    # Wait for the GPU only as long as this request's own budget allows.
    if not _GPU_LOCK.acquire(timeout=max(0.0, deadline - time.time())):
        logger.error("GPU still busy at the deadline for %s; guessing.", request.audio_filename)
        return _fallback(request.questions)
    try:
        return _predict_locked(request, audio_bytes, deadline)
    finally:
        _GPU_LOCK.release()


def _predict_locked(
    request: ASRQuestionRequestDto, audio_bytes: bytes, deadline: float
) -> ASRQuestionResponseDto:
    hotwords = (
        asr.hotwords_from_questions(request.questions) if asr.HOTWORDS_ENABLED else None
    )
    try:
        transcript = asr.transcribe_bytes(
            audio_bytes, request.audio_filename, hotwords=hotwords, deadline=deadline
        )
    except Exception:
        logger.exception(
            "Transcription failed for %s; returning guesses.", request.audio_filename
        )
        return _fallback(request.questions)

    answers: List[bool] = []
    evidence_start: List[Optional[float]] = []
    evidence_end: List[Optional[float]] = []

    try:
        answerer = get_answerer()
        results = answerer.answer_all(request.questions, transcript, deadline=deadline)
    except Exception:
        logger.exception(
            "Answering failed for %s; returning guesses.", request.audio_filename
        )
        return _fallback(request.questions)

    # Be defensive about the contract: the service scores by position and a
    # wrong-length response loses every question about the conversation.
    missing = guess(request.questions[len(results):]) if len(results) < len(request.questions) else []
    for question_index in range(len(request.questions)):
        if question_index < len(results):
            is_true, span = results[question_index][:2]
        else:
            is_true, span = missing[question_index - len(results)], None

        start, end = _clean_span(span) if is_true else (None, None)
        answers.append(bool(is_true))
        evidence_start.append(start)
        evidence_end.append(end)

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )


def _clean_span(span) -> tuple:
    """A well-formed ``(start, end)`` or ``(None, None)``: never half-filled or
    inverted, which the service would silently score as zero anyway."""
    try:
        if span is None:
            return None, None
        start, end = float(span[0]), float(span[1])
        if start != start or end != end or start < 0 or end <= start:
            return None, None
        return round(start, 3), round(end, 3)
    except Exception:
        return None, None


def _fallback(questions) -> ASRQuestionResponseDto:
    """A well-formed guess from the question text alone."""
    count = len(questions)
    return ASRQuestionResponseDto(
        answers=guess(questions),
        evidence_start=[None] * count,
        evidence_end=[None] * count,
    )
