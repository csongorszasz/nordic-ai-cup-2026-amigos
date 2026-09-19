"""Timing-only comparisons preserve every decision and the exact word occurrence."""

import copy

import pytest

from benchmark_alignment import alignment_gates, baseline_prediction, retime_prediction, word_anchor


WORDS = [
    {"word": " Good.", "start": 0.0, "end": 0.5},
    {"word": " Good.", "start": 3.0, "end": 3.5},
]
ROW = {
    "question_id": "q", "transcript_id": "s", "question": "Was it good?",
    "label": 1, "answer": True, "prediction": 1, "question_type": "positive",
    "span": [3.0, 3.5], "gold": [3.1, 3.4], "duration": 5.0,
    "quote": "Good.", "word_range": [1, 1],
}


@pytest.mark.parametrize("latency", [1.0, 300.0])
def test_cpu_quality_run_never_qualifies_as_gpu_feasibility(latency):
    report = {
        "device": "cpu", "conversation_failures": 0,
        "reasons": {"retimed": 5}, "max_estimated_combined_s": latency,
    }
    assert alignment_gates(report) == {"alignment_success": True, "feasibility_passed": False}


def test_gpu_gate_requires_successful_alignment_and_latency_headroom():
    report = {
        "device": "cuda", "conversation_failures": 0,
        "reasons": {"retimed": 5}, "max_estimated_combined_s": 30.0,
    }
    assert alignment_gates(report)["feasibility_passed"] is True
    assert alignment_gates({**report, "max_estimated_combined_s": 50.0})["feasibility_passed"] is False
    for changed in ({"conversation_failures": 1}, {"reasons": {"conversation_error_keep": 5}}):
        assert alignment_gates({**report, **changed}) == {
            "alignment_success": False, "feasibility_passed": False,
        }


def test_retiming_preserves_the_selected_repeated_occurrence():
    timed = [{**word, "start": index + 0.1, "end": index + 0.4} for index, word in enumerate(WORDS)]
    result = retime_prediction(ROW, WORDS, timed)
    assert result["span"] == [1.1, 1.4]
    assert result["word_range"] == [1, 1]
    assert result["alignment_reason"] == "retimed"
    assert result["answer"] is True
    assert result["quote"] == ROW["quote"]


def test_invalid_new_span_retains_corrected_incumbent_not_no_answer():
    timed = copy.deepcopy(WORDS)
    timed[1]["end"] = timed[1]["start"]
    result = retime_prediction(ROW, WORDS, timed)
    assert result["answer"] is True
    assert result["span"] == [3.2, 3.5]
    assert result["alignment_reason"] == "invalid_span_keep"


def test_ground_truth_cannot_gate_applicability_or_change_a_decision():
    timed = copy.deepcopy(WORDS)
    reference_changed = {**ROW, "label": 0, "gold": None, "question_type": "hard_negative"}
    assert retime_prediction(ROW, WORDS, timed)["span"] == retime_prediction(reference_changed, WORDS, timed)["span"]
    missed_positive = {**ROW, "answer": False, "prediction": 0, "span": None, "word_range": None}
    result = retime_prediction(missed_positive, WORDS, timed)
    assert result["answer"] is False and result["span"] is None
    assert result["alignment_reason"] == "base_no"


@pytest.mark.parametrize("anchor", [None, [], [0, 2], [-1, 0], [1, 0], [True, 1], [1.0, 1]])
def test_missing_or_invalid_frozen_anchor_fails_before_inference(anchor):
    with pytest.raises(ValueError, match="anchor"):
        word_anchor({**ROW, "word_range": anchor}, WORDS)


def test_anchor_must_match_both_quote_occurrence_and_original_times():
    for changed in ({**ROW, "word_range": [0, 0]}, {**ROW, "quote": "invented"}):
        with pytest.raises(ValueError, match="occurrence"):
            word_anchor(changed, WORDS)


def test_word_replacement_is_not_a_timing_only_comparison():
    timed = copy.deepcopy(WORDS)
    timed[1]["word"] = " Bad."
    with pytest.raises(ValueError, match="text/order"):
        retime_prediction(ROW, WORDS, timed)


def test_conversation_failure_fallback_keeps_all_reference_fields():
    result = baseline_prediction(ROW, "conversation_error_keep")
    assert result["span"] == [3.2, 3.5]
    assert result["prediction"] == ROW["prediction"]
    assert result["gold"] == ROW["gold"]
