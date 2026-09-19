"""Inverse-question prompts cannot copy the target or mask the wrong occurrence."""

from dataclasses import replace

import pytest

from answerers.evidence_units import build_evidence_units
from answerers.source_ranker import MODES, question_jobs, select_source, source_prompt


def fixture():
    words = [
        {"word": " " + word, "start": index * 0.5, "end": (index + 1) * 0.5, "seg_idx": 0}
        for index, word in enumerate("Normal. We discussed the dose. Normal. Take 100 mg.".split())
    ]
    units = build_evidence_units(words, 10.0)
    return words, units


def test_question_is_only_a_target_not_interpolated_into_any_encoder_prompt():
    words, units = fixture()
    question = "Does the unique target phrase appear in the consultation?"
    prompts, jobs = question_jobs([(question, units[:3])], words)
    assert all(question not in prompt for prompt in prompts)
    assert {job["target"] for job in jobs} == {question}
    assert len(jobs) == 3 * len(MODES)
    assert {job["mode"] for job in jobs} == set(MODES)


def test_context_ablation_uses_word_indices_not_first_string_match():
    words, units = fixture()
    occurrences = [unit for unit in units if unit.text == "Normal."]
    second = occurrences[1]
    prompt = source_prompt(second, words, "with_context")
    assert "Earlier context:\nWe discussed the dose." in prompt
    assert "Selected passage:\nNormal." in prompt
    assert "Later context:\nTake 100 mg." in prompt
    masked = source_prompt(second, words, "masked_source")
    assert "Selected passage:\n(selected passage withheld)" in masked
    assert "Normal." not in masked


def test_source_only_prompt_does_not_include_interpretation_context():
    words, units = fixture()
    unit = next(unit for unit in units if unit.text == "Take 100 mg.")
    prompt = source_prompt(unit, words, "source_only")
    assert "Take 100 mg." in prompt
    assert "Normal." not in prompt and "We discussed" not in prompt


def test_identical_prompts_are_encoded_once_without_merging_question_targets():
    words, units = fixture()
    selected = units[:2]
    prompts, jobs = question_jobs([("Question one?", selected), ("Question two?", selected)], words)
    assert len(prompts) < len(jobs)
    assert {job["target"] for job in jobs} == {"Question one?", "Question two?"}
    assert len([job for job in jobs if job["case_index"] == 0]) == 6
    assert len([job for job in jobs if job["case_index"] == 1]) == 6


def test_prompt_rejects_stale_text_or_crossed_context_ranges():
    words, units = fixture()
    with pytest.raises(ValueError, match="Source text"):
        source_prompt(replace(units[0], text="Invented."), words, "with_context")
    with pytest.raises(ValueError, match="ranges"):
        source_prompt(replace(units[0], context_first_word=10), words, "with_context")


def test_source_policies_are_fixed_and_ties_retain_the_incumbent():
    scores = [
        {"source_only": -2.0, "with_context": -1.0, "masked_source": -1.5},
        {"source_only": -1.0, "with_context": -1.0, "masked_source": -3.0},
    ]
    assert select_source(scores, "with_context") == 0
    assert select_source(scores, "source_only") == 1
    assert select_source(scores, "support_gain") == 1


def test_batched_floating_point_noise_cannot_break_an_incumbent_tie():
    scores = [
        {"source_only": -1.000001, "with_context": -1.000001, "masked_source": -2.0},
        {"source_only": -1.0, "with_context": -1.0, "masked_source": -2.0},
    ]
    for policy in ("source_only", "with_context", "support_gain"):
        assert select_source(scores, policy) == 0


@pytest.mark.parametrize("entry", [
    {}, {"source_only": -1.0}, {mode: float("nan") for mode in MODES},
    {mode: True for mode in MODES}, {mode: 1.0 for mode in MODES},
])
def test_partial_or_invalid_likelihoods_are_never_success_shaped(entry):
    with pytest.raises(ValueError, match="likelihood"):
        select_source([entry], "with_context")
