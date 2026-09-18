"""Ego-motion estimation between consecutive camera views.

The drone flies a straight line, so a ground object drifts through the source
frame at a roughly constant rate (about 58 source pixels per frame at the
supplied step). The world map needs that rate to carry tracks forward while the
camera is looking elsewhere, and to correct a track when it is re-observed.

Two estimators are provided:

* ``phase_correlation`` - the cheap, proven default. ``cv2.phaseCorrelate``
  measures the translation between two grayscale views.
* ``ecc`` - ``cv2.findTransformECC`` with a translation model. Slower but more
  robust to brightness changes and low-texture scenes.

Both report the shift of the *scene* relative to the frame in source pixels:
a positive ``dy`` means ground content moved down, which is what a forward
flying, downward-looking camera produces. When the camera itself was moved
between the two views, the caller supplies that known centre delta and it is
added back, because the measured content motion is ``ego - camera_delta``.

The estimator is stateless; smoothing and rejection live in the tracker so
that a bad measurement can never overwrite a good prior.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from dtos import IMAGE_WIDTH


@dataclass(slots=True)
class EgoMotionResult:
    """A per-frame scene shift measurement, in source pixels."""

    dx: float
    dy: float
    response: float
    accepted: bool
    method: str


class EgoMotionEstimator:
    """Measure per-frame scene shift between two grayscale views."""

    def __init__(
        self,
        method: str = "phase_correlation",
        min_response: float = 0.15,
        min_shift_y: float = 10.0,
        max_shift_y: float = 130.0,
        max_abs_shift_x: float = 40.0,
        ecc_iterations: int = 50,
        ecc_epsilon: float = 1e-4,
        ecc_working_width: int = 480,
    ):
        if method not in ("phase_correlation", "ecc"):
            raise ValueError(
                f"Unknown ego-motion method {method!r}; "
                f"expected 'phase_correlation' or 'ecc'"
            )
        self.method = method
        self.min_response = min_response
        self.min_shift_y = min_shift_y
        self.max_shift_y = max_shift_y
        self.max_abs_shift_x = max_abs_shift_x
        self.ecc_iterations = ecc_iterations
        self.ecc_epsilon = ecc_epsilon
        self.ecc_working_width = ecc_working_width

    # ------------------------------------------------------------------ #
    # Measurement
    # ------------------------------------------------------------------ #

    def _measure_phase_correlation(
        self, previous: np.ndarray, current: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        try:
            shift, response = cv2.phaseCorrelate(previous, current)
        except Exception:
            return None
        return float(shift[0]), float(shift[1]), float(response)

    def _measure_ecc(
        self, previous: np.ndarray, current: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        # ECC is O(n) in pixels and fragile on full 4K pairs; work on a
        # downscaled copy and scale the translation back up.
        height, width = current.shape[:2]
        scale = 1.0
        if width > self.ecc_working_width:
            scale = self.ecc_working_width / float(width)
            size = (self.ecc_working_width, max(1, int(round(height * scale))))
            previous = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
            current = cv2.resize(current, size, interpolation=cv2.INTER_AREA)

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_iterations,
            self.ecc_epsilon,
        )
        try:
            correlation, warp = cv2.findTransformECC(
                previous, current, warp, cv2.MOTION_TRANSLATION, criteria, None, 5
            )
        except cv2.error:
            return None

        # ``warp`` from findTransformECC maps the input (current) onto the
        # template (previous); its translation is therefore the negative of the
        # scene's motion from previous to current. The sign below was pinned
        # against phase correlation on real Helsinki frames so both estimators
        # report the same convention: positive dy means ground content moved
        # down.
        dx = float(warp[0, 2]) / scale
        dy = float(warp[1, 2]) / scale
        return dx, dy, float(correlation)

    def measure(
        self, previous: np.ndarray, current: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Return (dx, dy, response) in the units of the supplied images."""
        if self.method == "ecc":
            return self._measure_ecc(previous, current)
        return self._measure_phase_correlation(previous, current)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def estimate(
        self,
        previous_gray: Optional[np.ndarray],
        current_gray: Optional[np.ndarray],
        gap: int = 1,
        pixel_scale: float = 1.0,
        camera_delta: Tuple[float, float] = (0.0, 0.0),
        previous_pixel_scale: Optional[float] = None,
    ) -> EgoMotionResult:
        """Estimate the per-frame scene shift in source pixels.

        ``pixel_scale`` is source pixels per grayscale pixel for the current
        view. ``camera_delta`` is the known camera centre movement in source
        pixels between the two views. Estimation is rejected (``accepted``
        stays False) when the two views have different scales, when the
        correlation is weak, or when the result is outside the plausible
        forward-flight envelope. The caller keeps its prior in that case.
        """
        gap = max(1, int(gap))
        if previous_gray is None or current_gray is None:
            return EgoMotionResult(0.0, 0.0, 0.0, False, "none")
        if previous_gray.shape != current_gray.shape:
            return EgoMotionResult(0.0, 0.0, 0.0, False, self.method)
        if previous_pixel_scale is not None and not np.isclose(
            previous_pixel_scale, pixel_scale, rtol=1e-3
        ):
            # A level change changes the field of view; phase correlation would
            # report the zoom, not the ego-motion.
            return EgoMotionResult(0.0, 0.0, 0.0, False, self.method)

        measured = self.measure(previous_gray.astype(np.float32), current_gray.astype(np.float32))
        if measured is None:
            return EgoMotionResult(0.0, 0.0, 0.0, False, self.method)

        raw_dx, raw_dy, response = measured
        dx = (raw_dx * pixel_scale + camera_delta[0]) / gap
        dy = (raw_dy * pixel_scale + camera_delta[1]) / gap

        accepted = (
            response >= self.min_response
            and self.min_shift_y <= dy <= self.max_shift_y
            and abs(dx) <= self.max_abs_shift_x
        )
        return EgoMotionResult(dx, dy, response, accepted, self.method)


def default_pixel_scale(view_width: int) -> float:
    """Source pixels per view pixel if the view covered the whole frame."""
    if view_width <= 0:
        return float(IMAGE_WIDTH)
    return IMAGE_WIDTH / float(view_width)
