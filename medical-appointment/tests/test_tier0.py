"""Tier 0 hardening: candidate-cap diversity, cache keys, deadline fallback."""

import base64
import json
from pathlib import Path

import answer
import asr
import calibrate
import example
from dtos import ASRQuestionRequestDto
from utils import validate_response

FIXTURE = Path(__file__).parent / "fixtures" / "transcript_sample.json"


def test_cap_candidates_keeps_all_lengths():
    # 40 ranges of each length 1..6, limit far below the total.
    ranges = [(i, i + length) for length in range(1, 7) for i in range(40)]
    capped = answer._cap_candidates(ranges, limit=12)
    assert len(capped) == 12
    lengths = {j - i for i, j in capped}
    assert lengths == {1, 2, 3, 4, 5, 6}  # round-robin keeps every length


def test_cap_candidates_noop_when_small():
    ranges = [(0, 0), (0, 1)]
    assert answer._cap_candidates(ranges, limit=10) == ranges


def test_cache_path_keyed_by_content_and_config():
    a = asr.cache_path("conversation_sample_4.mp3", b"audio-one")
    b = asr.cache_path("conversation_sample_4.mp3", b"audio-two")
    again = asr.cache_path("conversation_sample_4.mp3", b"audio-one")
    assert a != b
    assert a == again
    assert "conversation_sample_4" in a.name and asr.config_hash() in a.name


def test_deadline_fallback_returns_valid_guess(monkeypatch):
    monkeypatch.setattr(asr, "transcribe_bytes", lambda *a, **k: json.loads(FIXTURE.read_text()))
    monkeypatch.setattr(example, "DEADLINE_S", 0.0)

    def should_not_run(*args, **kwargs):
        raise AssertionError("answer_question must be skipped past the deadline")

    monkeypatch.setattr(answer, "answer_question", should_not_run)

    request = ASRQuestionRequestDto(
        audio_base64=base64.b64encode(b"x").decode(),
        audio_filename="conversation_sample_17.mp3",
        questions=[f"Q{i}?" for i in range(10)],
    )
    response = example.predict(request)
    validate_response(response, 10)
    assert response.answers == [True] * 10
    assert response.evidence_start == [None] * 10


def test_calibrate_predict_and_score():
    records = [
        {"transcript_id": "s1", "label": 1, "p": 0.9, "guard_ok": True,
         "span": [1.0, 2.0], "gold": [1.0, 2.0]},
        {"transcript_id": "s1", "label": 0, "p": 0.1, "guard_ok": True,
         "span": [3.0, 4.0], "gold": None},
        {"transcript_id": "s2", "label": 1, "p": 0.2, "guard_ok": True,
         "span": [5.0, 6.0], "gold": [5.0, 6.0]},
    ]
    pred, span = calibrate.predict(records[0], tau=0.5)
    assert pred is True and tuple(span) == (1.0, 2.0)

    score, accuracy, mean_tiou = calibrate.evaluate(records, tau=0.5)
    # positives: s1 (yes, tIoU 1.0), s2 (p=0.2 < 0.5 -> no, tIoU 0); negative correct.
    assert abs(accuracy - 2 / 3) < 1e-9
    assert abs(mean_tiou - 0.5) < 1e-9
    assert abs(score - (0.4 * (2 / 3) + 0.6 * 0.5)) < 1e-9
