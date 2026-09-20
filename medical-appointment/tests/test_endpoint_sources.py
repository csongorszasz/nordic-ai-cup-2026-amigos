"""Endpoint selection preserves decisions, source identity, folds, and fallbacks."""

import copy

import pytest

import calibrate_endpoint_sources as calibration


def pair(tid):
    baseline = {
        "question_id": tid, "transcript_id": tid, "question": "Was it normal?",
        "answer": True, "prediction": 1, "label": 1, "question_type": "positive",
        "span": [1.2, 2.0], "gold": [1.2, 1.8], "word_range": [0, 1],
        "quote": "It is normal.", "duration": 5.0,
    }
    return baseline, {**baseline, "span": [1.0, 1.8], "ctc_reason": "aligned"}


def test_four_policies_use_already_calibrated_endpoints_without_new_offsets():
    base, aligned = pair("a")
    assert calibration.apply_policy(base, aligned, "incumbent")["span"] == [1.2, 2.0]
    assert calibration.apply_policy(base, aligned, "ctc_end")["span"] == [1.2, 1.8]
    assert calibration.apply_policy(base, aligned, "ctc_start")["span"] == [1.0, 2.0]
    assert calibration.apply_policy(base, aligned, "ctc_both")["span"] == [1.0, 1.8]
    assert calibration.fit_policy([(base, aligned)]) == "ctc_end"


def test_applicability_does_not_consult_reference_fields():
    base, aligned = pair("a")
    changed = {**base, "label": 0, "gold": None, "question_type": "hard_negative"}
    assert calibration.apply_policy(base, aligned, "ctc_end")["span"] == (
        calibration.apply_policy(changed, aligned, "ctc_end")["span"]
    )


def test_ineligible_negative_and_disjoint_clock_cases_keep_the_incumbent():
    base, aligned = pair("a")
    skipped = {**aligned, "span": base["span"], "ctc_reason": "ineligible_keep"}
    assert calibration.apply_policy(base, skipped, "ctc_both")["span"] == base["span"]
    disjoint = {**aligned, "span": [3.0, 3.5]}
    for policy in ("ctc_start", "ctc_end", "ctc_both"):
        selected = calibration.apply_policy(base, disjoint, policy)
        assert selected["span"] == base["span"] and selected["endpoint_reason"] == "disjoint_clocks_keep"
    no = {**base, "answer": False, "prediction": 0, "span": None, "word_range": None}
    assert calibration.apply_policy(no, {**no, "ctc_reason": "base_no"}, "ctc_end")["span"] is None


def test_tied_policies_prefer_the_simpler_incumbent():
    base, aligned = pair("a")
    assert calibration.fit_policy([(base, {**aligned, "span": base["span"]})]) == "incumbent"


def test_policy_selection_excludes_all_held_out_and_demonstration_conversations(monkeypatch):
    pairs = [pair(f"s{index}") for index in range(6)]
    seen = []

    def fit(training):
        seen.append({base["transcript_id"] for base, _ in training})
        return "ctc_end"

    monkeypatch.setattr(calibration, "fit_policy", fit)
    predictions, folds = calibration.cross_validate(pairs, {"s0"}, 13)
    assert len(predictions) == 5 and len({row["question_id"] for row in predictions}) == 5
    for training, fold in zip(seen, folds):
        assert training == set(fold["training_tids"])
        assert training.isdisjoint(set(fold["held_out_tids"]) | {"s0"})
    assert all(row["span"] == [1.2, 1.8] for row in predictions)


def test_mismatched_source_occurrence_and_incomplete_inputs_fail():
    base, aligned = pair("a")
    with pytest.raises(ValueError, match="occurrence"):
        calibration.apply_policy(base, {**aligned, "word_range": [5, 6]}, "ctc_end")
    with pytest.raises(ValueError, match="at least five"):
        calibration.cross_validate([(base, aligned)], set(), 13)


def test_saved_fallback_spans_cannot_be_silently_recalibrated():
    base, aligned = pair("a")
    raw = {**base, "span": [1.0, 2.0]}
    assert calibration.matched_pairs([raw], [aligned])[0][0]["span"] == base["span"]
    corrupt = {**aligned, "ctc_reason": "ineligible_keep"}
    with pytest.raises(ValueError, match="fallback changed"):
        calibration.matched_pairs([raw], [corrupt])
    original = copy.deepcopy(base)
    calibration.apply_policy(base, aligned, "ctc_end")
    assert base == original
