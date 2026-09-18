"""Verbatim quote -> word timestamp alignment."""

from answerers.align import align_quote, align_span, text_between

WORDS = [
    {"word": " Morning,", "start": 0.0, "end": 0.5},
    {"word": " Dr", "start": 0.6, "end": 0.9},
    {"word": " Fabricius.", "start": 1.0, "end": 1.4},
    {"word": " The", "start": 1.5, "end": 1.7},
    {"word": " dose", "start": 1.8, "end": 2.0},
    {"word": " is", "start": 2.1, "end": 2.2},
    {"word": " 100", "start": 2.3, "end": 2.5},
    {"word": " mg.", "start": 2.6, "end": 2.9},
]


def test_align_span_found():
    assert align_span(WORDS, "Morning, Dr Fabricius.") == (0.0, 1.4)
    assert align_span(WORDS, "The dose is 100 mg.") == (1.5, 2.9)


def test_align_quote_returns_word_indices():
    assert align_quote(WORDS, "dose is 100") == (1.8, 2.5, 4, 6)


def test_align_span_case_insensitive_and_whitespace():
    assert align_span(WORDS, "the   DOSE is") == (1.5, 2.2)


def test_align_span_paraphrase_fails():
    assert align_span(WORDS, "one hundred milligrams") is None
    assert align_span(WORDS, "") is None


def test_text_between():
    assert text_between(WORDS, 1.5, 2.9) == "The dose is 100 mg."
