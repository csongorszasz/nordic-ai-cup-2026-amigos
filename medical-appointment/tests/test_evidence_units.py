"""The citation hierarchy and its shortlist only consume runtime inputs."""

import copy
from dataclasses import asdict

import pytest

from answerers.evidence_units import (
    build_evidence_units, incumbent_unit, retrieval_scores, shortlist_evidence,
)
from answerers.passages import build_sentence_atoms, build_sentences


def words_for(text):
    return [
        {"word": " " + token, "start": index * 0.4, "end": (index + 1) * 0.4, "seg_idx": 0}
        for index, token in enumerate(text.split())
    ]


def test_short_answer_is_preserved_without_changing_legacy_sentence_merging():
    words = words_for("Was the examination normal? Yes. We can discuss the prescription.")
    atoms = build_sentence_atoms(words)
    assert [atom.text for atom in atoms] == [
        "Was the examination normal?", "Yes.", "We can discuss the prescription.",
    ]
    assert all(atom.text != "Yes." for atom in build_sentences(words))
    units = build_evidence_units(words, 10)
    reply = next(unit for unit in units if unit.text == "Yes.")
    assert reply.word_range() == (4, 4)
    assert "Was the examination normal?" in reply.context_text
    assert reply.start == words[4]["start"]
    assert reply.end == words[4]["end"]
    assert reply.context_first_word < reply.first_word < reply.context_last_word


def test_clause_and_procedure_episode_coexist_with_the_result_statement():
    words = words_for("Let me listen. All right. Breathe in, then out. Nothing abnormal.")
    units = build_evidence_units(words, 10)
    assert any(unit.text == "Breathe in," and "clause" in unit.families for unit in units)
    result = next(unit for unit in units if unit.text == "Nothing abnormal.")
    episode = next(unit for unit in units if unit.first_word == 0 and unit.last_word == len(words) - 1)
    assert "episode" in episode.families
    assert result.word_range() != episode.word_range()
    assert len({unit.word_range() for unit in units}) == len(units)


def test_identical_words_remain_distinct_occurrences():
    words = words_for("Normal. Other things changed. Normal.")
    units = build_evidence_units(words, 10)
    occurrences = [unit for unit in units if unit.text == "Normal."]
    assert len(occurrences) == 2
    assert occurrences[0].index != occurrences[1].index
    assert occurrences[0].span() != occurrences[1].span()


def test_gold_metadata_cannot_change_candidate_geometry_or_retrieval():
    words = words_for("Take 100 mg. Do not take 200 mg. No. Continue after meals.")
    original = copy.deepcopy(words)
    decorated = [
        {**word, "gold": [90, 91], "label": 0, "question_type": "off_topic", "filename": "sample_4"}
        for word in words
    ]
    left = build_evidence_units(words, 20)
    right = build_evidence_units(decorated, 20)
    assert [asdict(unit) for unit in left] == [asdict(unit) for unit in right]
    question = "Was 100 mg after meals prescribed?"
    assert retrieval_scores(question, left) == retrieval_scores(question, right)
    assert words == original
    assert [asdict(unit) for unit in shortlist_evidence(question, left, 20)] == [
        asdict(unit) for unit in shortlist_evidence(question, right, 20)
    ]


def test_shortlist_preserves_incumbent_once_and_retrieves_distant_occurrences():
    words = words_for("Good morning. " + "We discussed other matters. " * 20 + "Take 100 mg after meals.")
    duration = words[-1]["end"]
    units = build_evidence_units(words, duration)
    incumbent = incumbent_unit(words, duration, [0.2, 0.8], [0, 1])
    selected = shortlist_evidence("Was 100 mg after meals prescribed?", units, duration, incumbent=incumbent)
    assert selected[0] is incumbent
    assert selected[0].resolved_span(duration) == [0.2, 0.8]
    assert len(selected) <= 33
    assert len({tuple(unit.resolved_span(duration)) for unit in selected}) == len(selected)
    assert any(unit.first_word > 24 and "100" in unit.text for unit in selected)
    assert any("episode" in unit.families for unit in selected)
    assert any("clause" in unit.families for unit in selected)


def test_retrieval_keeps_short_clinical_terms_negation_and_numbers():
    words = words_for("The left arm hurts. The right leg hurts. No 100 mg. Take 200 mg.")
    units = build_evidence_units(words, 20)
    scores = retrieval_scores("Does the left arm hurt?", units)
    left = next(unit for unit in units if unit.text == "The left arm hurts.")
    right = next(unit for unit in units if unit.text == "The right leg hurts.")
    assert scores[left.index] > scores[right.index]
    assert retrieval_scores("No 100 mg?", units) != retrieval_scores("Take 200 mg?", units)


def test_episodes_do_not_expand_across_a_long_gap():
    words = words_for("First statement. Second statement.")
    for word in words[2:]:
        word["start"] += 40
        word["end"] += 40
    units = build_evidence_units(words, 50)
    assert not any("episode" in unit.families for unit in units)


def test_zero_duration_atoms_are_logged_not_used_as_invalid_intervals(caplog):
    words = [{"word": " Yes.", "start": 0.0, "end": 0.0}]
    assert build_evidence_units(words, 1) == []
    assert "zero-duration" in caplog.text


@pytest.mark.parametrize("span,anchor", [
    ([0.0, 0.0], [0, 0]), ([0.0, 20.0], [0, 0]), ([False, 1.0], [0, 0]),
    ([0.0, 1.0], [True, 1]), ([0.0, 1.0], [1, 0]),
])
def test_invalid_incumbent_fails_explicitly(span, anchor):
    with pytest.raises(ValueError, match="incumbent"):
        incumbent_unit(words_for("Yes. No."), 5, span, anchor)


def test_missing_or_nonfinite_retrieval_scores_are_rejected():
    units = build_evidence_units(words_for("Yes. No."), 5)
    for scores in ({}, {unit.index: float("nan") for unit in units}):
        with pytest.raises(ValueError, match="Retrieval scores"):
            shortlist_evidence("Was it normal?", units, 5, scores=scores)


def test_question_order_cannot_change_the_candidates():
    words = words_for("Take 100 mg. Do not take 200 mg. The result is normal.")
    units = build_evidence_units(words, 20)
    questions = ["Was 100 mg advised?", "Was 200 mg advised?", "Was the result normal?"]
    forward = {question: [unit.index for unit in shortlist_evidence(question, units, 20)] for question in questions}
    reverse = {question: [unit.index for unit in shortlist_evidence(question, units, 20)] for question in reversed(questions)}
    assert forward == reverse
