"""Lexical differences are review flags, never invented word-error-rate labels."""

from probe_asr_hints import changes


def transcript(*words):
    return {"words": [{"word": " " + word} for word in words]}


def test_drug_spelling_changes_and_dose_negation_flags_are_separate():
    before = transcript("No", "panadil", "50", "mg.")
    after = transcript("Panodil", "100", "mg.")
    result = changes(before, after, ["Panodil", "Ibumetin"])
    assert result["term_counts"]["Panodil"] == {"before": 0, "after": 1}
    assert result["term_counts"]["Ibumetin"] == {"before": 0, "after": 0}
    assert result["number_tokens_before"] == ["50"]
    assert result["number_tokens_after"] == ["100"]
    assert result["negation_tokens_before"] == ["no"]
    assert result["negation_tokens_after"] == []
    assert result["edit_blocks"]
    assert "wer" not in result and "improved" not in result
