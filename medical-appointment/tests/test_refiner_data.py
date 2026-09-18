"""Anchored refiner examples against the E4B probe records."""

from pathlib import Path

import pytest

from answerers import refiner_data


def test_build_refiner_examples_are_well_formed():
    path = Path(refiner_data.DEFAULT_LLM_QUESTIONS)
    if not path.exists():
        pytest.skip("E4B probe question records are not present")

    examples, skipped = refiner_data.build_refiner_examples(str(path), window=24)
    assert examples, f"no examples built; skipped={skipped}"
    for example in examples[:25]:
        assert example["label"] == "support"
        first, last = example["span_words"]
        assert 0 <= first <= last < len(example["passage_words"])
        assert example["first_word"] <= example["gold_word_range"][0]
        assert example["gold_word_range"][1] <= example["last_word"]
        assert example["first_word"] <= example["llm_word_range"][0]
        assert example["llm_word_range"][1] <= example["last_word"]
        assert 0.0 <= example["target_tiou"] <= 1.0
        assert len(example["passage_words"]) == example["last_word"] - example["first_word"] + 1
