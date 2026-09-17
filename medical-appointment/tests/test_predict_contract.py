"""The protocol contract: response shape and the never-raise guarantee."""

import base64
import json
from pathlib import Path

import answer
import asr
import example
from dtos import ASRQuestionRequestDto
from utils import validate_response

FIXTURE = Path(__file__).parent / "fixtures" / "transcript_sample.json"
QUESTIONS = [f"Question number {i}?" for i in range(10)]


def make_request(questions=None):
    return ASRQuestionRequestDto(
        audio_base64=base64.b64encode(b"not-real-audio").decode(),
        audio_filename="conversation_sample_17.mp3",
        questions=list(questions if questions is not None else QUESTIONS),
    )


def load_fixture():
    return json.loads(FIXTURE.read_text())


def test_response_contract(monkeypatch):
    monkeypatch.setattr(asr, "transcribe_bytes", lambda *a, **k: load_fixture())
    monkeypatch.setattr(
        answer,
        "answer_question",
        lambda q, words, windows: (True, (0.5, 1.5)) if q.endswith("0?") else (False, None),
    )

    response = example.predict(make_request())

    assert len(response.answers) == len(QUESTIONS)
    assert all(isinstance(a, bool) for a in response.answers)
    assert len(response.evidence_start) == len(QUESTIONS)
    assert len(response.evidence_end) == len(QUESTIONS)
    for answer_value, start, end in zip(
        response.answers, response.evidence_start, response.evidence_end
    ):
        if answer_value:
            assert start is not None and end is not None and start <= end
        else:
            assert start is None and end is None

    validate_response(response, len(QUESTIONS))


def test_never_raises_when_answering_fails(monkeypatch):
    monkeypatch.setattr(asr, "transcribe_bytes", lambda *a, **k: load_fixture())

    def explode(*args, **kwargs):
        raise RuntimeError("verifier blew up")

    monkeypatch.setattr(answer, "answer_question", explode)

    response = example.predict(make_request())

    validate_response(response, len(QUESTIONS))
    assert response.answers == [True] * len(QUESTIONS)
    assert response.evidence_start == [None] * len(QUESTIONS)
    assert response.evidence_end == [None] * len(QUESTIONS)


def test_fallback_when_transcription_fails(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("asr blew up")

    monkeypatch.setattr(asr, "transcribe_bytes", explode)

    response = example.predict(make_request())

    validate_response(response, len(QUESTIONS))
    assert response.answers == [True] * len(QUESTIONS)
    assert response.evidence_start == [None] * len(QUESTIONS)
