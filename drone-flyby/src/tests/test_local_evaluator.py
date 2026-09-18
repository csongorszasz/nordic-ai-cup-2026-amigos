"""Pure helpers of the local evaluation harness."""

import local_evaluator
from local_evaluator import _percentile, size_bin_recall

CAMERA_FRAMES = [{"frame": 0, "level": 0, "center": [1920, 1080], "region": [0, 0, 3840, 2160]}]


def test_percentile_nearest_rank():
    ordered = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(ordered, 50) == 30.0
    assert _percentile(ordered, 99) == 50.0
    assert _percentile([], 50) == 0.0


def test_size_bin_boundaries_have_no_gap(monkeypatch):
    # At L0 the transmitted scale is 960/3840 = 0.25, so a source box of side
    # 28/30/61 px becomes 7.0/7.5/15.25 px on the wire.
    boxes = [[0, 0, 28, 28], [0, 0, 30, 30], [0, 0, 61, 61]]
    annotations = [{"object_id": "tank", "bbox": box} for box in boxes]
    predictions = {0: [{"object_id": "tank", "bbox": box, "confidence": 1.0} for box in boxes]}

    monkeypatch.setattr(local_evaluator, "frame_numbers", lambda scene: [0])
    monkeypatch.setattr(local_evaluator, "load_annotations", lambda frame, scene: annotations)

    bins = size_bin_recall("scene", predictions, CAMERA_FRAMES)

    assert bins["4-7px"] == (1, 1)      # exactly 7.0 px
    assert bins["8-15px"] == (1, 1)     # 7.5 px
    assert bins[">15px"] == (1, 1)      # 15.25 px
