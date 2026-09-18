"""Unit tests for ego-motion estimation."""

import cv2
import numpy as np
import pytest

from core.ego_motion import EgoMotionEstimator, default_pixel_scale
from utils import load_frame


@pytest.fixture(scope="module")
def grayscale_frame() -> np.ndarray:
    frame = load_frame(0, scene="helsinki")
    return cv2.cvtColor(
        cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))


def test_phase_correlation_reports_content_shift(grayscale_frame):
    estimator = EgoMotionEstimator(
        method="phase_correlation", min_shift_y=-1000, max_shift_y=1000,
        max_abs_shift_x=1000,
    )
    current = _shift(grayscale_frame, 0, 20)

    result = estimator.estimate(grayscale_frame, current, pixel_scale=1.0)

    assert result.accepted
    assert result.dy == pytest.approx(20.0, abs=1.0)
    assert result.dx == pytest.approx(0.0, abs=1.0)


def test_pixel_scale_multiplies_the_measurement(grayscale_frame):
    estimator = EgoMotionEstimator(
        method="phase_correlation", min_shift_y=-1000, max_shift_y=1000,
        max_abs_shift_x=1000,
    )
    current = _shift(grayscale_frame, 0, 10)

    result = estimator.estimate(grayscale_frame, current, pixel_scale=4.0)

    assert result.dy == pytest.approx(40.0, abs=2.0)


def test_ecc_agrees_with_phase_correlation(grayscale_frame):
    current = _shift(grayscale_frame, -15, -20)
    kwargs = dict(min_shift_y=-1000, max_shift_y=1000, max_abs_shift_x=1000)

    phase = EgoMotionEstimator(method="phase_correlation", **kwargs).estimate(
        grayscale_frame, current, pixel_scale=1.0
    )
    ecc = EgoMotionEstimator(method="ecc", **kwargs).estimate(
        grayscale_frame, current, pixel_scale=1.0
    )

    assert ecc.accepted
    assert ecc.dx == pytest.approx(phase.dx, abs=2.0)
    assert ecc.dy == pytest.approx(phase.dy, abs=2.0)


def test_camera_delta_is_added_back(grayscale_frame):
    # The camera moved down 10 px, so the content only appears to move 10 px
    # when the true ego-motion is 20 px.
    estimator = EgoMotionEstimator(
        method="phase_correlation", min_shift_y=-1000, max_shift_y=1000,
        max_abs_shift_x=1000,
    )
    current = _shift(grayscale_frame, 0, 10)

    result = estimator.estimate(
        grayscale_frame, current, pixel_scale=1.0, camera_delta=(0.0, 10.0)
    )

    assert result.dy == pytest.approx(20.0, abs=1.5)


def test_gap_divides_the_measurement(grayscale_frame):
    estimator = EgoMotionEstimator(
        method="phase_correlation", min_shift_y=-1000, max_shift_y=1000,
        max_abs_shift_x=1000,
    )
    current = _shift(grayscale_frame, 0, 40)

    result = estimator.estimate(grayscale_frame, current, gap=4, pixel_scale=1.0)

    assert result.dy == pytest.approx(10.0, abs=1.0)


def test_scale_change_is_rejected(grayscale_frame):
    estimator = EgoMotionEstimator(
        method="phase_correlation", min_shift_y=-1000, max_shift_y=1000,
        max_abs_shift_x=1000,
    )
    current = _shift(grayscale_frame, 0, 20)

    result = estimator.estimate(
        grayscale_frame,
        current,
        pixel_scale=2.0,
        previous_pixel_scale=4.0,
    )

    assert not result.accepted


def test_implausibly_large_shift_is_rejected(grayscale_frame):
    estimator = EgoMotionEstimator(method="phase_correlation")
    current = _shift(grayscale_frame, 0, 500)

    result = estimator.estimate(grayscale_frame, current, pixel_scale=1.0)

    assert not result.accepted


def test_default_pixel_scale_for_full_frame_and_l0():
    assert default_pixel_scale(3840) == pytest.approx(1.0)
    assert default_pixel_scale(960) == pytest.approx(4.0)
