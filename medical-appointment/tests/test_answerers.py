"""The answerer protocol: factory selection and the frozen legacy wrapper."""

import json
from pathlib import Path

import answer
import pytest
import windows as windows_module
from answerers import get_answerer
from answerers.legacy import LegacyAnswerer

FIXTURE = Path(__file__).parent / "fixtures" / "transcript_sample.json"


def load_transcript():
    return json.loads(FIXTURE.read_text())


def fake_scorer(premises, hypothesis, batch_size=16):
    scores = []
    for premise in premises:
        low = premise.lower()
        if "100 mg" in low:
            scores.append(0.9)
        elif "asthma" in low:
            scores.append(0.05)
        else:
            scores.append(0.3)
    return scores


def test_factory_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("MEDAPP_ANSWERER", raising=False)
    assert get_answerer().name == "legacy"


def test_factory_rejects_unknown(monkeypatch):
    monkeypatch.setenv("MEDAPP_ANSWERER", "nope")
    with pytest.raises(ValueError):
        get_answerer()


def test_factory_builds_modernbert_placeholder():
    assert get_answerer("modernbert").name == "modernbert"


def test_modernbert_placeholder_refuses_until_trained():
    answerer = get_answerer("modernbert")
    with pytest.raises(NotImplementedError):
        answerer.answer_all(["a?"], {"words": []})


def test_legacy_wrapper_matches_direct_answer_question(monkeypatch):
    monkeypatch.setattr(answer.nli, "score", fake_scorer)
    transcript = load_transcript()
    words = transcript["words"]
    windows = windows_module.windows_from_transcript(transcript)
    questions = [
        "Was the daily dose 100 mg?",
        "Does the patient have asthma?",
        "Was the prescribed dose 200 mg daily?",
    ]

    direct = [
        answer.answer_question(question, words, windows)
        for question in questions
    ]
    wrapped = LegacyAnswerer().answer_all(questions, transcript)

    assert wrapped == [(bool(is_true), span) for is_true, span in direct]


def test_legacy_deadline_returns_guesses(monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("answer_question must be skipped past the deadline")

    monkeypatch.setattr(answer, "answer_question", should_not_run)
    transcript = load_transcript()

    answers = LegacyAnswerer().answer_all(
        ["a?", "b?"], transcript, deadline=0.0
    )

    assert answers == [(True, None), (True, None)]
