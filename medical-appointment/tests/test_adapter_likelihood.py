"""Likelihood reranking uses grounded candidates and a fixed finite-score rule."""

import pytest

from rerank_adapter_quotes import canonical_quote, prefer_adapter


def test_quote_is_canonicalized_at_the_actual_proposed_time():
    words = [
        {"word": " Yes.", "start": 1.0, "end": 1.5},
        {"word": " Yes.", "start": 10.0, "end": 10.5},
    ]
    assert canonical_quote(words, "yes", [10.2, 10.5], 20.0) == "Yes."
    assert canonical_quote(words, "yes", [5.0, 5.5], 20.0) is None


def test_strict_finite_mean_likelihood_comparison():
    assert prefer_adapter(1.0, 0.8)
    assert not prefer_adapter(1.0, 1.0)
    assert not prefer_adapter(1.0, 1.2)
    with pytest.raises(ValueError, match="finite"):
        prefer_adapter(float("nan"), 1.0)
