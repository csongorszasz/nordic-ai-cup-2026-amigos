"""Timing transfer cannot silently replace text, doses, negation or occurrences."""

import math

import pytest

from answerers.retime import retime_words, word_unit_map


def words(*texts):
    return [{"word": text, "start": 50, "end": 51, "probability": 0.9} for text in texts]


def units(*texts):
    return [
        {"text": text, "start_time": float(index + 1), "end_time": float(index + 2)}
        for index, text in enumerate(texts)
    ]


def test_punctuation_normalization_preserves_original_text_and_fields():
    original = words("Well,", " don't", " take", " 100", " mg.")
    result = retime_words(original, units("well", "don't", "take", "100", "mg"), 10)
    assert [item["word"] for item in result] == [item["word"] for item in original]
    assert all(item["probability"] == 0.9 for item in result)
    assert [(item["start"], item["end"]) for item in result] == [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    ]
    assert all(item["start"] == 50 for item in original)


def test_subwords_and_different_word_groupings_keep_index_anchors():
    original = words(" The", " follow-", "up", " is fine.")
    assert word_unit_map(original, ["The", "followup", "is", "fine"]) == [
        (0, 0), (1, 1), (1, 1), (2, 3),
    ]
    result = retime_words(original, units("The", "followup", "is", "fine"), 10)
    assert result[1]["start"] == result[2]["start"] == 2
    assert result[-1]["start"] == 3 and result[-1]["end"] == 5


def test_punctuation_only_tokens_use_boundaries_not_stale_asr_times():
    result = retime_words(words('"', " Hello", '!"'), units("Hello"), 10)
    assert [(item["start"], item["end"]) for item in result] == [(1, 1), (1, 2), (2, 2)]


def test_repeated_words_preserve_their_ordered_occurrence():
    result = retime_words(words(" Yes.", " Yes."), units("Yes", "Yes"), 10)
    assert result[0]["start"] == 1
    assert result[1]["start"] == 2


@pytest.mark.parametrize("source,target", [
    ([" 100", " mg"], ["2100", "mg"]),
    ([" do", " not"], ["do"]),
    ([" no", " change"], ["change", "no"]),
    ([" ibuprofen"], ["paracetamol"]),
    (["..."], []),
])
def test_text_changes_cannot_masquerade_as_timing_changes(source, target):
    with pytest.raises(ValueError, match="letters or numbers"):
        word_unit_map(words(*source), target)


@pytest.mark.parametrize("start,end", [
    (-1, 2), (2, 1), (1, 11), (math.nan, 2), (1, math.inf), (True, 2), ("1", 2),
])
def test_invalid_timestamps_are_explicit_failures(start, end):
    with pytest.raises(ValueError, match="timestamps"):
        retime_words(words("hi"), [{"text": "hi", "start_time": start, "end_time": end}], 10)


def test_out_of_order_units_are_not_repaired_silently():
    aligned = units("one", "two")
    aligned[1]["start_time"] = 1.5
    with pytest.raises(ValueError, match="ordered"):
        retime_words(words("one", " two"), aligned, 10)


@pytest.mark.parametrize("duration", [0, -1, math.inf, math.nan, True, None])
def test_invalid_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        retime_words(words("hi"), units("hi"), duration)
