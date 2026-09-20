"""Verifier internals that do not need the model: label-order detection."""

from types import SimpleNamespace

from verifier import nli


def test_entail_index_from_id2label():
    config = SimpleNamespace(
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={},
    )
    assert nli._find_entail_index(config) == 0


def test_entail_index_from_label2id():
    config = SimpleNamespace(
        id2label={},
        label2id={"contradiction": 0, "neutral": 1, "entailment": 2},
    )
    assert nli._find_entail_index(config) == 2


def test_entail_index_defaults_when_unknown():
    config = SimpleNamespace(id2label={}, label2id={})
    assert nli._find_entail_index(config) == 0


def test_score_empty_premises_returns_empty():
    assert nli.score([], "hypothesis") == []
