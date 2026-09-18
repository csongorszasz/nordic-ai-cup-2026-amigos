"""ModernBERT token-span mapping and the score-aware decoder (no model)."""

from answerers.span_utils import (
    LABEL_NOT_MENTIONED,
    LABEL_SUPPORT,
    char_span_to_words,
    pick_supported_candidate,
    psupport_accept,
    score_aware_accept,
    target_token_span,
    token_span_to_char,
    word_char_spans,
)

# [CLS] q0 q1 [SEP] w0 w1 w2 w3 w4 w5 [SEP]
OFFSETS = [
    (0, 0), (0, 4), (5, 9), (0, 0),
    (0, 3), (4, 8), (9, 11), (12, 15), (16, 18), (19, 24), (0, 0),
]
SEQ_IDS = [None, 0, 0, None, 1, 1, 1, 1, 1, 1, None]


def test_word_char_spans():
    text, spans = word_char_spans(["the", "dose", "is", "100", "mg"])
    assert text == "the dose is 100 mg"
    assert spans == [(0, 3), (4, 8), (9, 11), (12, 15), (16, 18)]


def test_target_token_span_selects_passage_tokens():
    # Words "dose" (4,8) .. "100" (12,15) -> passage token indices 5..7.
    assert target_token_span(OFFSETS, SEQ_IDS, 4, 15) == (5, 7)


def test_target_token_span_ignores_question_side():
    # A question-side char range must never resolve to a question token.
    start, end = target_token_span(OFFSETS, SEQ_IDS, 0, 4)
    assert start >= 4 and end >= 4


def test_target_token_span_falls_back_to_nearest():
    start, end = target_token_span(OFFSETS, SEQ_IDS, 100, 101)
    assert start == end == 9


def test_token_span_to_char():
    assert token_span_to_char(OFFSETS, SEQ_IDS, 5, 7) == (4.0, 15.0)
    assert token_span_to_char(OFFSETS, SEQ_IDS, 0, 5) is None  # question token
    assert token_span_to_char(OFFSETS, SEQ_IDS, 7, 5) == (4.0, 15.0)  # swapped


def test_char_span_to_words():
    text, spans = word_char_spans(["the", "dose", "is", "100", "mg", "daily"])
    assert char_span_to_words(spans, 4.0, 15.0) == (1, 3)
    assert char_span_to_words(spans, 100.0, 101.0) is None


def test_score_aware_threshold():
    # q = 0.36 -> tau ~ 0.325; a missed positive is doubly costly so tau < 0.5.
    assert score_aware_accept(0.33, 0.36)
    assert not score_aware_accept(0.32, 0.36)
    # Better span -> lower threshold.
    assert score_aware_accept(0.21, 1.0)
    # No span quality -> plain 0.5, strict.
    assert not score_aware_accept(0.5, 0.0)


def test_psupport_threshold():
    assert psupport_accept(0.09, 0.08)
    assert not psupport_accept(0.08, 0.08)  # strict


def test_pick_supported_candidate_prefers_best_tiou():
    candidates = [
        {"label": LABEL_NOT_MENTIONED, "p_support": 0.1, "expected_tiou": 0.9},
        {"label": LABEL_SUPPORT, "p_support": 0.8, "expected_tiou": 0.4},
        {"label": LABEL_SUPPORT, "p_support": 0.7, "expected_tiou": 0.7},
    ]
    assert pick_supported_candidate(candidates) == 2
    assert pick_supported_candidate([candidates[0]]) is None
