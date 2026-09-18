"""Sliding-window passage construction (no model)."""

from answerers.passages import (
    STRIDE_WORDS,
    WINDOW_WORDS,
    build_passages,
    contains,
    overlap_word_range,
)


def make_words(count):
    return [
        {"word": f" w{i}", "start": float(i), "end": float(i) + 0.5}
        for i in range(count)
    ]


def test_any_short_range_is_covered():
    words = make_words(200)
    passages = build_passages(words, window=28, stride=14)
    for first in range(0, 200, 7):
        for length in range(1, 15):
            last = first + length - 1
            if last >= 200:
                continue
            assert any(contains(p, (first, last)) for p in passages)


def test_tail_is_anchored():
    words = make_words(50)
    passages = build_passages(words, window=28, stride=14)
    assert contains(passages[-1], (49, 49))
    assert passages[-1].last_word == 49


def test_empty_words():
    assert build_passages([]) == []


def test_passage_spans_match_words():
    words = make_words(40)
    passages = build_passages(words, window=10, stride=5)
    for passage in passages:
        assert passage.span() == (
            words[passage.first_word]["start"],
            words[passage.last_word]["end"],
        )


def test_overlap_word_range():
    words = make_words(10)
    assert overlap_word_range(words, 2.2, 4.2) == (2, 4)
    assert overlap_word_range(words, 100.0, 101.0) is None


def test_served_window_always_covers_its_bound():
    # The chosen serving config guarantees any span up to W - S + 1 words is
    # fully contained regardless of alignment. Guard against a silent regression.
    window, stride = WINDOW_WORDS, STRIDE_WORDS
    bound = window - stride + 1
    words = make_words(4 * window)
    passages = build_passages(words, window=window, stride=stride)
    for first in range(0, 2 * window):
        for length in range(1, bound + 1):
            last = first + length - 1
            assert any(contains(p, (first, last)) for p in passages), (
                first, last, window, stride,
            )
