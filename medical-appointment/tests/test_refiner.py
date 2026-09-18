"""Boundary refiner pure helpers (no torch)."""

from answerers.refiner import (
    BoundaryRefiner,
    apply_deltas,
    build_window_passage,
    joint_token_span,
    marked_window_layout,
    relative_word_span,
    relative_word_span_from_spans,
    window_bounds,
)

WORDS = [
    {"word": " a", "start": 0.0, "end": 0.5},
    {"word": " b", "start": 0.5, "end": 1.0},
    {"word": " c", "start": 1.0, "end": 1.5},
    {"word": " d", "start": 1.5, "end": 2.0},
    {"word": " e", "start": 2.0, "end": 2.5},
]


def test_window_bounds_clamp():
    assert window_bounds(5, 2, 3, context=1) == (1, 4)
    assert window_bounds(5, 0, 0, context=3) == (0, 3)
    assert window_bounds(5, 4, 4, context=3) == (1, 4)


def test_build_window_passage():
    passage = build_window_passage(WORDS, 1, 2, context=1)
    assert passage.first_word == 0
    assert passage.last_word == 3
    assert passage.span() == (0.0, 2.0)
    assert passage.text == "a b c d"


def test_relative_word_span_maps_tokens_to_words():
    offsets = [(0, 0), (0, 1), (2, 3), (4, 5), (6, 7), (0, 0)]
    sequence_ids = [None, 1, 1, 1, 1, None]
    assert relative_word_span(
        ["a", "b", "c", "d"], offsets, sequence_ids, 3, 4
    ) == (2, 3)


def test_relative_word_span_rejects_non_passage_tokens():
    offsets = [(0, 0), (0, 1), (2, 3), (0, 0)]
    sequence_ids = [None, 1, 1, None]
    assert relative_word_span(["a", "b"], offsets, sequence_ids, 0, 0) is None


def test_refiner_without_checkpoint_returns_none(monkeypatch):
    monkeypatch.delenv("MEDAPP_LLM_REFINER", raising=False)
    refiner = BoundaryRefiner(checkpoint=None)
    assert refiner.refine("q", WORDS, 1, 2) is None


def test_marked_window_layout_marks_the_anchor():
    text, spans = marked_window_layout(["a", "b", "c", "d"], 1, 2)
    assert text == "a [EVIDENCE] b c [/EVIDENCE] d"
    assert spans == [(0, 1), (13, 14), (15, 16), (29, 30)]
    assert text[spans[1][0]:spans[1][1]] == "b"
    assert text[spans[3][0]:spans[3][1]] == "d"


def test_marked_window_layout_closes_at_the_tail():
    text, spans = marked_window_layout(["a", "b"], 0, 1)
    assert text.endswith("[/EVIDENCE]")
    assert len(spans) == 2
    assert text[spans[0][0]:spans[0][1]] == "a"


def test_joint_token_span_pairing_and_cap():
    start_scores = [0.0, 5.0, 1.0, 9.0, 0.0]
    end_scores = [0.0, 1.0, 7.0, 0.5, 0.0]
    sequence_ids = [None, 1, 1, 1, None]
    assert joint_token_span(start_scores, end_scores, sequence_ids) == (1, 2)
    assert joint_token_span(start_scores, end_scores, sequence_ids, max_words=1) == (3, 3)
    assert joint_token_span([0.0] * 5, [0.0] * 5, [None, None, None, None, None]) is None


def test_relative_word_span_from_spans_uses_given_spans():
    offsets = [(0, 0), (0, 1), (2, 3), (4, 5), (0, 0)]
    sequence_ids = [None, 1, 1, 1, None]
    word_spans = [(0, 1), (2, 3), (4, 5)]
    assert relative_word_span_from_spans(
        word_spans, offsets, sequence_ids, 3, 3
    ) == (2, 2)


def test_apply_deltas_trims_and_expands():
    assert apply_deltas(10, 15, 0.0, 0.0, 0, 100) == (10, 15)
    assert apply_deltas(10, 15, 2.4, -3.6, 0, 100) == (8, 11)
    assert apply_deltas(10, 15, -2.0, 4.0, 0, 100) == (12, 19)


def test_apply_deltas_never_inverts_the_span():
    assert apply_deltas(10, 11, 0.0, -5.0, 0, 100) == (10, 10)


def test_apply_deltas_clamps_to_window():
    assert apply_deltas(10, 15, 50.0, 50.0, 5, 20) == (5, 20)
