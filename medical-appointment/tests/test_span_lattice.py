"""Local candidates retain the baseline and cannot use labels to choose a region."""

import math

import pytest

from answerers.span_lattice import local_span_lattice, word_token_indices


WORDS = [
    {"word": f" w{index}", "start": float(index), "end": index + 0.5}
    for index in range(8)
]


def test_baseline_is_unique_zero_prior_candidate():
    result = local_span_lattice(WORDS, [3, 4], [3.2, 4.5], 8.0, context_words=2)
    assert (result["first_word"], result["last_word"]) == (1, 6)
    assert result["pairs"][0] == [2, 3]
    assert result["spans"][0] == [3.2, 4.5]
    assert result["priors"][0] == 0
    assert all(prior < 0 for prior in result["priors"][1:])
    assert len({tuple(span) for span in result["spans"]}) == len(result["spans"])
    assert all(first <= last for first, last in result["pairs"])
    assert all(0 <= start < end <= 8 for start, end in result["spans"])


def test_short_span_collapse_uses_existing_runtime_semantics():
    result = local_span_lattice(
        [{"word": "Good", "start": 0.0, "end": 0.16}], [0, 0], [0.0, 0.16], 0.16,
    )
    assert result["spans"] == [[0.0, 0.16]]
    assert result["priors"] == [0.0]


def test_extra_annotation_fields_cannot_change_candidate_geometry():
    original = local_span_lattice(WORDS, [3, 4], [3.2, 4.5], 8)
    annotated = [{**word, "gold": [50, 60], "label": 0} for word in WORDS]
    assert local_span_lattice(annotated, [3, 4], [3.2, 4.5], 8) == original


def test_pooling_uses_only_exact_passage_tokens():
    offsets = [(0, 0), (0, 4), (0, 0), (0, 2), (2, 4), (5, 7), (0, 0)]
    sequences = [None, 0, None, 1, 1, 1, None]
    assert word_token_indices(offsets, sequences, [(0, 4), (5, 7)]) == [[3, 4], [5]]
    with pytest.raises(ValueError, match="do not truncate"):
        word_token_indices(offsets, sequences, [(8, 10)])


@pytest.mark.parametrize("anchor,span,context", [
    (None, [3.2, 4.5], 24), ([3, 4], None, 24),
    ([True, 4], [3.2, 4.5], 24), ([4, 3], [3.2, 4.5], 24),
    ([0, 8], [3.2, 4.5], 24), ([3, 4], [math.nan, 4.5], 24),
    ([3, 4], [4.5, 4.5], 24), ([3, 4], [3.2, 4.5], -1),
])
def test_invalid_candidate_inputs_fail(anchor, span, context):
    with pytest.raises(ValueError):
        local_span_lattice(WORDS, anchor, span, 8, context)
