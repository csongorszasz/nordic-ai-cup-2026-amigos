"""Only valid new aligner spans can receive training-fold boundary correction."""

import pytest

import calibrate_alignment as calibration


def row(tid, reason="retimed", answer=True):
    return {
        "question_id": f"{tid}_{reason}", "transcript_id": tid,
        "label": 1, "answer": answer, "prediction": int(answer), "question_type": "positive",
        "gold": [0.2, 1.0], "span": [0.0, 1.0] if answer else None,
        "duration": 10.0, "alignment_reason": reason,
    }


def test_fallbacks_are_not_shifted_a_second_time():
    records = [row("a"), row("b", "invalid_span_keep"), row("c", "base_no", False)]
    result = calibration.apply_alignment_offsets(records, (0.2, 0.0))
    assert result[0]["span"] == [0.2, 1.0]
    assert result[1]["span"] == records[1]["span"]
    assert result[2]["span"] is None
    assert records[0]["span"] == [0.0, 1.0]


def test_calibration_applicability_is_not_a_gold_label_gate():
    original = row("a")
    changed = {**original, "label": 0, "gold": None, "question_type": "hard_negative"}
    assert calibration.apply_alignment_offsets([original], (0.2, 0.0))[0]["span"] == (
        calibration.apply_alignment_offsets([changed], (0.2, 0.0))[0]["span"]
    )


def test_every_question_is_retained_and_fitting_excludes_held_out_and_demos(monkeypatch):
    rows = [row(f"s{index}") for index in range(6)]
    rows += [row(f"s{index}", "invalid_span_keep") for index in range(6)]
    seen = []

    def fit(training, grid):
        assert all(item["alignment_reason"] == "retimed" for item in training)
        seen.append({item["transcript_id"] for item in training})
        return 0.2, 0.0

    monkeypatch.setattr(calibration, "fit_offsets", fit)
    predictions, folds = calibration.cross_validate_alignment(rows, {"s0"}, 13)
    assert len(predictions) == 10
    assert len({item["question_id"] for item in predictions}) == 10
    for fitted, fold in zip(seen, folds):
        assert fitted.isdisjoint(set(fold["held_out_tids"]) | {"s0"})
        assert fitted == set(fold["training_tids"])
    assert all(
        item["span"] == [0.0, 1.0] for item in predictions
        if item["alignment_reason"] == "invalid_span_keep"
    )


def test_too_few_conversations_cannot_produce_five_fold_evidence():
    with pytest.raises(ValueError, match="at least five"):
        calibration.cross_validate_alignment([row("a")], set(), 13)
