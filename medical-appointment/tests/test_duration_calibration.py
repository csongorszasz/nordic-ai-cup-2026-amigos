"""Duration calibration is label-independent at inference and grouped when fitted."""

import pytest

import calibrate_duration as calibration
from answerers.boundaries import adjusted_span


def row(tid, label):
    return {
        "question_id": f"{tid}_{label}", "transcript_id": tid,
        "label": label, "prediction": label, "answer": bool(label),
        "question_type": "positive" if label else "hard_negative",
        "gold": [1.2, 3.0] if label else None, "span": [1.0, 3.0] if label else None,
        "duration": 5.0,
    }


def test_baseline_is_exactly_the_existing_transform():
    for span in ([1.0, 3.0], [0.0, 0.16], [4.8, 5.0]):
        assert calibration.transform_span(span, calibration.BASELINE, 5) == adjusted_span(span, (0.2, 0), 5)
    assert calibration.BASELINE in calibration.GRID
    assert len(calibration.GRID) == 64


def test_fractional_trimming_scales_with_input_span_not_labels():
    assert calibration.transform_span([1, 5], (0, 0, 0.1, 0.05), 8) == [1.4, 4.8]
    assert calibration.transform_span(None, calibration.BASELINE, 8) is None
    original = row("s", 1)
    changed = {**original, "gold": [3, 4], "label": 0}
    assert calibration.apply_parameters([original], calibration.BASELINE)[0]["span"] == (
        calibration.apply_parameters([changed], calibration.BASELINE)[0]["span"]
    )


def test_grouped_fit_keeps_all_questions_and_excludes_demos(monkeypatch):
    rows = [row(f"s{i}", label) for i in range(6) for label in (0, 1)]
    observed = []

    def fit(training):
        observed.append({item["transcript_id"] for item in training})
        return calibration.BASELINE

    monkeypatch.setattr(calibration, "fit_parameters", fit)
    predictions, folds = calibration.cross_validate(rows, {"s0"}, 13)
    assert len(predictions) == len({item["question_id"] for item in predictions}) == 10
    for training, fold in zip(observed, folds):
        assert training.isdisjoint(set(fold["held_out_tids"]) | {"s0"})
    assert all(item["span"] is None for item in predictions if not item["answer"])


def test_nonfinite_parameters_fail():
    with pytest.raises(ValueError, match="finite"):
        calibration.transform_span([1, 2], (0.2, 0, float("nan"), 0), 5)
