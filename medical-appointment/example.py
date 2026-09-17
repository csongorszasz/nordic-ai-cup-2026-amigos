"""The model behind ``/predict``.

Transcribe once per conversation with local ASR (word timestamps kept), build
phrase-level windows, then answer each question by NLI entailment and return the
tightest supporting word range. Nothing here calls a cloud API.

``predict`` never raises: one exception would score all ten of a conversation's
questions wrong, so every failure falls back to a well-formed guess.
"""

import logging
import os
import time
from typing import List, Optional

import answer as answer_module
import asr
import windows as windows_module
from capture import maybe_capture
from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio

logger = logging.getLogger(__name__)

# Load the models at import time: the first inference is the slowest and there
# is no warm-up budget. Skipped in tests via MEDAPP_SKIP_WARMUP=1.
if os.environ.get("MEDAPP_SKIP_WARMUP") != "1":
    try:
        asr.warm_up()
        from verifier import nli as _nli

        _nli.warm_up()
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
    audio_bytes = decode_audio(request.audio_base64)

    try:
        transcript = asr.transcribe_bytes(audio_bytes, request.audio_filename)
        words = transcript.get("words", [])
        windows = windows_module.windows_from_transcript(transcript)
    except Exception:
        logger.exception(
            "Transcription failed for %s; returning guesses.", request.audio_filename
        )
        return _fallback(len(request.questions))

    answers: List[bool] = []
    evidence_start: List[Optional[float]] = []
    evidence_end: List[Optional[float]] = []

    for question in request.questions:
        try:
            is_true, span = answer_module.answer_question(question, words, windows)
        except Exception:
            logger.exception("Answering failed; guessing for: %s", question)
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
