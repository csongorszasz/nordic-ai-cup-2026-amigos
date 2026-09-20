"""Offsets use training groups only and always retain legal intervals."""

from calibrate_spans import adjusted_span, cross_validate, fit_offsets
from answerers.boundaries import OffsetCalibration


def rows():
    return [
        {
            "question_id": f"q{index}", "transcript_id": f"s{index}",
            "label": 1, "question_type": "positive", "prediction": 1,
            "answer": True, "span": [1.0, 2.0], "gold": [1.2, 2.0],
            "duration": 4.0,
        }
        for index in range(5)
    ]


def test_offset_fit_recovers_consistent_displacement():
    assert fit_offsets(rows()) == (0.2, 0.0)


def test_calibration_groups_and_demonstrations_never_leak():
    baseline, candidate, folds = cross_validate(rows(), {"s0"}, n_folds=2)
    assert len(baseline) == len(candidate) == 4
    for fold in folds:
        assert set(fold["train_tids"]).isdisjoint(fold["held_out_tids"])
        assert "s0" not in fold["train_tids"] + fold["held_out_tids"]
    assert all(row["span"] == [1.2, 2.0] for row in candidate)


def test_infeasible_offsets_keep_baseline_and_audio_bounds():
    assert adjusted_span([0.1, 0.2], (0.4, -0.4)) == [0.1, 0.2]
    assert adjusted_span([0.1, 0.9], (-0.4, 0.4), duration=1.0) == [0.0, 1.0]


def test_runtime_calibration_matches_the_offline_transform():
    calibration = OffsetCalibration(0.2, 0.0, "asr", "model", "revision", "base", "hash")
    assert calibration.apply([1.0, 2.0], 4.0) == (1.2, 2.0)
    calibration.validate_context(
        asr_config_hash="asr", model="model", revision="revision", variant="base"
    )


def test_calibration_refuses_different_model_or_asr():
    import pytest

    calibration = OffsetCalibration(0.2, 0.0, "asr", "model", "revision", "base", "hash")
    with pytest.raises(ValueError, match="does not match"):
        calibration.validate_context(
            asr_config_hash="other-asr", model="model", revision="revision", variant="base"
        )


def test_fine_grid_is_explicit_and_keeps_baseline_comparable():
    grid = [(0.15, 0.0), (0.2, 0.0), (0.25, 0.0)]
    assert fit_offsets(rows(), grid) == (0.2, 0.0)
    _, candidate, folds = cross_validate(rows(), set(), 2, 13, grid)
    assert all(fold["offsets"] == [0.2, 0.0] for fold in folds)
    assert all(row["span"] == [1.2, 2.0] for row in candidate)


def test_tracked_release_artifact_matches_the_incumbent_contract():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "calibration" / "span_offset_base.json"
    calibration = OffsetCalibration.load(path)
    assert (calibration.start_offset_s, calibration.end_offset_s) == (0.2, 0.0)
    assert calibration.model == "google/gemma-4-26b-a4b-it"
    assert calibration.revision == "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
    assert calibration.variant == "base"
    assert calibration.asr_config_hash == "e75a7f6e"
