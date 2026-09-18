"""Wire-protocol validation rules the evaluator also enforces."""

import pytest
from pydantic import ValidationError

from dtos import DroneFlybyPredictionDto, RequestedViewDto


def _prediction(**overrides):
    values = {"object_id": "tank", "bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.5}
    values.update(overrides)
    return DroneFlybyPredictionDto(**values)


def test_rejects_zero_area_box():
    with pytest.raises(ValidationError):
        _prediction(bbox=[0.1, 0.1, 0.1, 0.2])


def test_rejects_unknown_object_id():
    with pytest.raises(ValidationError):
        _prediction(object_id="car")


def test_rejects_non_finite_confidence():
    with pytest.raises(ValidationError):
        _prediction(confidence=float("nan"))


def test_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        _prediction(confidence=1.5)


def test_accepts_full_frame_box():
    prediction = _prediction(bbox=[0.0, 0.0, 1.0, 1.0])
    assert list(prediction.bbox) == [0.0, 0.0, 1.0, 1.0]


def test_requested_view_rejects_float_coordinates():
    with pytest.raises(ValidationError):
        RequestedViewDto(resolution_level=1, center_x=1.0, center_y=2)
