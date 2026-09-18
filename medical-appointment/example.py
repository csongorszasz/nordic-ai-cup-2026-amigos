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
import os
import time
from typing import List, Optional

import asr
from answerers import get_answerer
from capture import maybe_capture
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio

logger = logging.getLogger(__name__)

# Leave a margin under the 60 s request budget: stop answering new questions and
# return guesses once this much wall-clock has elapsed.
DEADLINE_S = float(os.environ.get("MEDAPP_DEADLINE_S", "50"))

# Load the models at import time: the first inference is the slowest and there
# is no warm-up budget. Skipped in tests via MEDAPP_SKIP_WARMUP=1.
if os.environ.get("MEDAPP_SKIP_WARMUP") != "1":
    try:
        asr.warm_up()
        answerer = get_answerer()
        warm_up = getattr(answerer, "warm_up", None)
        if warm_up is not None:
            warm_up()
    except Exception:  # pragma: no cover - startup environment issue
        logger.exception("Model warm-up failed; will load lazily on first request.")


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation. Never raises."""
    started = time.time()
    try:
        response = _predict(request)
    except Exception:
        logger.exception(
            "predict failed for %s; returning guesses.", request.audio_filename
        )
        response = _fallback(len(request.questions))

    maybe_capture(request, response, time.time() - started)
    return response


def _predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    started = time.time()
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
        answerer = get_answerer()
        results = answerer.answer_all(
            request.questions, transcript, deadline=started + DEADLINE_S
        )
    except Exception:
        logger.exception(
            "Answering failed for %s; returning guesses.", request.audio_filename
        )
        return _fallback(len(request.questions))

    # Be defensive about the contract: the service scores by position and a
    # wrong-length response loses every question about the conversation.
    for question_index in range(len(request.questions)):
        if question_index < len(results):
            is_true, span = results[question_index]
        else:
            is_true, span = True, None

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
        answers=[True] * count,
        evidence_start=[None] * count,
        evidence_end=[None] * count,
    )
