"""Decode rules for the ModernBERT answerer (no model)."""

from answerers.modernbert import decode_question
from answerers.passages import Passage
from answerers.span_utils import LABEL_NOT_MENTIONED, LABEL_SUPPORT


class FakeModule:
    def predicted_span_seconds(self, passage, words, offsets, sequence_ids, start, end):
        return passage.span()


def make_candidate(passage, p_support, label, expected_tiou=0.5):
    return {
        "passage": passage,
        "result": {
            "offsets": [], "sequence_ids": [],
            "start_index": 0, "end_index": 0,
        },
        "label": label,
        "p_support": p_support,
        "expected_tiou": expected_tiou,
    }


PASSAGE_A = Passage(0, 0, 1, 0.0, 1.0, "a")
PASSAGE_B = Passage(1, 2, 3, 2.0, 3.0, "b")


def test_psupport_picks_max_and_accepts():
    candidates = [
        make_candidate(PASSAGE_A, 0.2, LABEL_NOT_MENTIONED),
        make_candidate(PASSAGE_B, 0.9, LABEL_NOT_MENTIONED),
    ]
    answer, span, info = decode_question(
        candidates, [], FakeModule(), mode="psupport", tau=0.5
    )
    assert answer is True
    assert span == PASSAGE_B.span()
    assert info["decided_by"] == "support"


def test_psupport_below_tau_returns_span_for_calibration():
    candidates = [make_candidate(PASSAGE_A, 0.3, LABEL_NOT_MENTIONED)]
    answer, span, info = decode_question(
        candidates, [], FakeModule(), mode="psupport", tau=0.5
    )
    assert answer is False and span is None
    assert info["span"] == PASSAGE_A.span()
    assert info["decided_by"] == "below_threshold"


def test_score_aware_needs_support_label():
    candidates = [make_candidate(PASSAGE_A, 0.95, LABEL_NOT_MENTIONED)]
    answer, span, info = decode_question(
        candidates, [], FakeModule(), mode="score_aware"
    )
    assert answer is False and span is None
    assert info["decided_by"] == "no_support"


def test_score_aware_accepts_with_good_span():
    candidates = [
        make_candidate(PASSAGE_A, 0.8, LABEL_SUPPORT, expected_tiou=1.0),
    ]
    answer, span, _ = decode_question(
        candidates, [], FakeModule(), mode="score_aware"
    )
    assert answer is True and span == PASSAGE_A.span()


def test_no_candidates_answers_no():
    answer, span, info = decode_question([], [], FakeModule())
    assert answer is False and span is None
    assert info["decided_by"] == "no_candidates"
