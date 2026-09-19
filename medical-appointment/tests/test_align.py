"""Verbatim quote -> word timestamp alignment."""

from answerers.align import align_quote, align_quote_matches, align_span, text_between

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


def test_align_span_rejects_substrings_inside_words():
    words = [{"word": " 2100", "start": 0.0, "end": 0.5}]
    assert align_span(words, "100") is None


def test_align_span_allows_omitted_edge_punctuation():
    assert align_span(WORDS, "Morning") == (0.0, 0.5)
    assert align_span(WORDS, "100 mg") == (2.3, 2.9)


def test_alignment_does_not_split_numbers_or_drug_names():
    words = [
        {"word": " 3.5", "start": 0.0, "end": 0.5},
        {"word": " amoxicillin-clavulanate", "start": 0.6, "end": 1.0},
        {"word": " 100", "start": 1.1, "end": 1.5},
    ]
    assert align_span(words, "5") is None
    assert align_span(words, "amoxicillin") is None
    assert align_span(words, "10 0") is None


def test_align_quote_can_be_scoped_to_an_occurrence():
    words = WORDS + [
        {"word": " The", "start": 3.0, "end": 3.2},
        {"word": " dose", "start": 3.3, "end": 3.5},
        {"word": " is", "start": 3.6, "end": 3.7},
        {"word": " 100", "start": 3.8, "end": 4.0},
        {"word": " mg.", "start": 4.1, "end": 4.4},
    ]
    matches = align_quote_matches(words, "The dose is 100 mg.")
    assert len(matches) == 2
    assert align_quote(words, "The dose is 100 mg.", first_word=8) == (
        3.0, 4.4, 8, 12
    )


def test_text_between():
    assert text_between(WORDS, 1.5, 2.9) == "The dose is 100 mg."
