"""Refinement cannot change decisions or escape the supplied local region."""

from refine_quotes import decode_refinement


WORDS = [
    {"word": " The", "start": 0.0, "end": 0.3},
    {"word": " dose", "start": 0.4, "end": 0.8},
    {"word": " is", "start": 0.9, "end": 1.1},
    {"word": " 100", "start": 1.2, "end": 1.5},
    {"word": " mg.", "start": 1.6, "end": 1.9},
]
TRANSCRIPT = {"words": WORDS, "duration": 2.0}
CASE = {"first": 0, "last": 4, "row": {"span": [0.0, 1.9], "quote": "The dose is 100 mg."}}


def test_keep_and_bad_quotes_preserve_baseline():
    assert decode_refinement({"keep": True}, CASE, TRANSCRIPT) == ([0.2, 1.9], "kept")
    assert decode_refinement(None, CASE, TRANSCRIPT) == ([0.2, 1.9], "missing")
    assert decode_refinement({"quote": "not here"}, CASE, TRANSCRIPT) == ([0.2, 1.9], "unaligned")


def test_unique_refinement_aligns_inside_the_region():
    assert decode_refinement({"quote": "100 mg."}, CASE, TRANSCRIPT) == ([1.4, 1.9], "refined")
    narrow = {**CASE, "last": 2}
    assert decode_refinement({"quote": "100 mg."}, narrow, TRANSCRIPT) == ([0.2, 1.9], "unaligned")
