"""Answerer orchestration with a fake NLI scorer (no model)."""

import json
from pathlib import Path

import answer
import windows as windows_module

FIXTURE = Path(__file__).parent / "fixtures" / "transcript_sample.json"


def load_transcript():
    data = json.loads(FIXTURE.read_text())
    return data["words"], windows_module.windows_from_transcript(data)


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


def test_localizes_the_supporting_phrase(monkeypatch):
    monkeypatch.setattr(answer.nli, "score", fake_scorer)
    words, windows = load_transcript()

    is_true, span = answer.answer_question(
        "Was the daily dose 100 mg?", words, windows
    )

    assert is_true is True
    assert span == (1.8, 2.4)  # the words "100 mg", not the whole clause
    # tighter than the clause window it came from
    assert span[1] - span[0] < windows[1].end - windows[1].start


def test_below_threshold_answers_no(monkeypatch):
    monkeypatch.setattr(
        answer.nli, "score", lambda premises, hypothesis, batch_size=16: [0.1] * len(premises)
    )
    words, windows = load_transcript()

    assert answer.answer_question("Does the patient have diabetes?", words, windows) == (
        False,
        None,
    )


def test_numeric_guard_vetoes_near_miss(monkeypatch):
    monkeypatch.setattr(
        answer.nli, "score", lambda premises, hypothesis, batch_size=16: [0.9] * len(premises)
    )
    words, windows = load_transcript()

    # The evidence says 100 mg; the question asserts 200 mg.
    assert answer.answer_question(
        "Was the prescribed dose 200 mg daily?", words, windows
    ) == (False, None)


def test_empty_conversation_answers_no():
    assert answer.answer_question("Anything?", [], []) == (False, None)
