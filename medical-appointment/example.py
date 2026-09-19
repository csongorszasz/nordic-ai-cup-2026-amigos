"""The model behind ``/predict``.

Transcribe once per conversation with local ASR (word timestamps kept), then
hand the questions to the answerer selected by ``MEDAPP_ANSWERER`` (see
``answerers/``). The default is the frozen NLI pipeline; a task-trained
cross-encoder can be A/B'd behind the same contract. Nothing here calls a cloud
API.

``predict`` never raises: one exception would score all ten of a conversation's
questions wrong, so every failure falls back to a well-formed guess.
"""

import logging
import multiprocessing
import os
import time
from typing import List, Optional

import asr
from answerers import Answerer, get_answerer
from answerers.base import normalize_answer
from capture import maybe_capture
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio, validate_response

logger = logging.getLogger(__name__)

# Leave a margin under the 60 s request budget: stop answering new questions and
# return guesses once this much wall-clock has elapsed.
DEADLINE_S = float(os.environ.get("MEDAPP_DEADLINE_S", "50"))
USE_WORKER = os.environ.get("MEDAPP_INFERENCE_WORKER", "0") == "1"
_worker = None


def warm_models() -> None:
    asr.warm_up()
    answerer = get_answerer()
    warm_up = getattr(answerer, "warm_up", None)
    if warm_up is not None:
        warm_up()


def get_worker():
    global _worker
    if _worker is None:
        import atexit
        from inference_worker import InferenceWorker

        _worker = InferenceWorker()
        atexit.register(_worker.close)
    return _worker


def predict(
    request: ASRQuestionRequestDto, *, answerer: Optional[Answerer] = None
) -> ASRQuestionResponseDto:
    """Answer every question about one conversation. Never raises."""
    started = time.monotonic()
    try:
        if USE_WORKER and answerer is None:
            payload = get_worker().predict(
                request.model_dump(), deadline=started + DEADLINE_S
            )
            response = (
                ASRQuestionResponseDto.model_validate(payload)
                if payload is not None else _fallback(len(request.questions))
            )
        else:
            response = _predict(request, answerer=answerer, deadline=started + DEADLINE_S)
        validate_response(response, len(request.questions))
    except Exception:
        logger.exception(
            "predict failed for %s; returning guesses.", request.audio_filename
        )
        response = _fallback(len(request.questions))

    maybe_capture(request, response, time.monotonic() - started)
    return response


def _predict(
    request: ASRQuestionRequestDto, *, answerer: Optional[Answerer] = None,
    deadline: Optional[float] = None,
) -> ASRQuestionResponseDto:
    started = time.monotonic()
    deadline = min(deadline, started + DEADLINE_S) if deadline is not None else started + DEADLINE_S
    if started >= deadline:
        logger.warning("Request deadline expired before transcription.")
        return _fallback(len(request.questions))
    audio_bytes = decode_audio(request.audio_base64)

    try:
        transcript = asr.transcribe_bytes(audio_bytes, request.audio_filename)
    except Exception:
        logger.exception(
            "Transcription failed for %s; returning guesses.", request.audio_filename
        )
        return _fallback(len(request.questions))

    answers: List[bool] = []
    evidence_start: List[Optional[float]] = []
    evidence_end: List[Optional[float]] = []

    try:
        answerer = answerer if answerer is not None else get_answerer()
        results = answerer.answer_all(
            request.questions, transcript, deadline=deadline
        )
    except Exception:
        logger.exception(
            "Answering failed for %s; returning guesses.", request.audio_filename
        )
        return _fallback(len(request.questions))

    # Be defensive about the contract: the service scores by position and a
    # wrong-length response loses every question about the conversation.
    for question_index in range(len(request.questions)):
        entry = results[question_index] if question_index < len(results) else None
        is_true, span = normalize_answer(
            entry, duration=transcript.get("duration"),
            context=f"{request.audio_filename} question {question_index}",
        )

        answers.append(bool(is_true))
        evidence_start.append(span[0] if span is not None else None)
        evidence_end.append(span[1] if span is not None else None)

    return ASRQuestionResponseDto(
        answers=answers,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
    )


def _fallback(count: int) -> ASRQuestionResponseDto:
    """A well-formed guess when the expensive half fails outright."""
    return ASRQuestionResponseDto(
        answers=[False] * count,
        evidence_start=[None] * count,
        evidence_end=[None] * count,
    )


if (
    os.environ.get("MEDAPP_SKIP_WARMUP") != "1"
    and multiprocessing.current_process().name == "MainProcess"
):
    try:
        if USE_WORKER:
            get_worker().warm_up()
        else:
            warm_models()
    except Exception:
        logger.exception("Model warm-up failed.")
        if os.environ.get("MEDAPP_REQUIRE_WARMUP", "0") == "1":
            raise
