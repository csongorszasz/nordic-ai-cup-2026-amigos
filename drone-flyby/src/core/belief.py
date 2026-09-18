"""Coarse belief field over the source frame for information-gain planning.

The field is the camera planner's picture of what it knows. It holds two things
per grid cell:

* how deeply the cell has been observed (``-1`` never, ``0/1/2`` the deepest
  resolution level at which it was seen), which drives **exploration**;
* an implicit per-cell object interest derived from the tracker's tracks, which
  drives **verification**.

The planner selects the next view by maximising a value of information: the
expected reduction in uncertainty about the scene, penalised by travel distance.
This is the architecture the research recommends to replace a fixed sweep: it
still guarantees coverage (unobserved cells always compete), but it can spend a
frame confirming a known, ambiguous object instead of walking an empty row.

The field is deliberately deterministic and dependency-light so it can be unit
tested and replayed. It never predicts object existence by itself; that
evidence comes from the tracker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.interfaces import TrackBelief
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH

# A cell is a square of this many source pixels. 120 px gives a 32x18 grid,
# fine enough to describe where the camera has and has not looked without
# making the field expensive to update.
DEFAULT_CELL_SIZE = 120


@dataclass(slots=True)
class CandidateView:
    """A scored next-view proposal."""

    level: int
    center_x: int
    center_y: int
    value: float
    kind: str  # "verify" or "explore"


class BeliefField:
    """Coverage and interest grid over the 3840x2160 source frame."""

    def __init__(
        self,
        cell_size: int = DEFAULT_CELL_SIZE,
        explore_weight: float = 1.0,
        verify_weight: float = 1.2,
        travel_weight: float = 0.35,
        min_track_existence: float = 0.20,
    ):
        self.cell_size = max(16, int(cell_size))
        self.columns = int(math.ceil(IMAGE_WIDTH / self.cell_size))
        self.rows = int(math.ceil(IMAGE_HEIGHT / self.cell_size))
        self.observed_level = np.full((self.rows, self.columns), -1, dtype=np.int8)
        self.last_seen_frame = np.full((self.rows, self.columns), -1, dtype=np.int32)

        self.explore_weight = float(explore_weight)
        self.verify_weight = float(verify_weight)
        self.travel_weight = float(travel_weight)
        self.min_track_existence = float(min_track_existence)

    # ------------------------------------------------------------------ #
    # Grid helpers
    # ------------------------------------------------------------------ #

    def cell_center(self, row: int, column: int) -> Tuple[float, float]:
        return (
            (column + 0.5) * self.cell_size,
            (row + 0.5) * self.cell_size,
        )

    def _cell_range(self, minimum: float, maximum: float, limit: int) -> Tuple[int, int]:
        """Inclusive-ish cell index range covering [minimum, maximum)."""
        start = int(max(0, math.floor(minimum / self.cell_size)))
        end = int(min(limit - 1, math.ceil(maximum / self.cell_size) - 1))
        if end < start:
            end = start
        return start, end

    def _region_slices(self, source_region: Sequence[int]) -> Tuple[slice, slice]:
        x1, y1, x2, y2 = (int(v) for v in source_region)
        column_start, column_end = self._cell_range(x1, x2, self.columns)
        row_start, row_end = self._cell_range(y1, y2, self.rows)
        return (
            slice(row_start, row_end + 1),
            slice(column_start, column_end + 1),
        )

    # ------------------------------------------------------------------ #
    # Updates
    # ------------------------------------------------------------------ #

    def mark_observed(
        self, source_region: Sequence[int], level: int, frame_index: int
    ) -> None:
        """Record that ``source_region`` was observed at ``level`` this frame."""
        rows, columns = self._region_slices(source_region)
        window = self.observed_level[rows, columns]
        np.maximum(window, level, out=window)
        self.last_seen_frame[rows, columns] = frame_index

    def reset(self) -> None:
        self.observed_level.fill(-1)
        self.last_seen_frame.fill(-1)

    # ------------------------------------------------------------------ #
    # Coverage queries
    # ------------------------------------------------------------------ #

    def coverage_fraction(self, level: int) -> float:
        """Fraction of cells observed at least at ``level``."""
        return float(np.count_nonzero(self.observed_level >= level)) / float(
            self.rows * self.columns
        )

    def unobserved_fraction(self, source_region: Sequence[int], level: int) -> float:
        """Fraction of cells in a region not yet observed at ``level``."""
        rows, columns = self._region_slices(source_region)
        window = self.observed_level[rows, columns]
        total = window.size
        if total == 0:
            return 0.0
        return float(np.count_nonzero(window < level)) / float(total)

    def is_fully_covered(self, level: int) -> bool:
        return bool(np.all(self.observed_level >= level))

    # ------------------------------------------------------------------ #
    # Information value
    # ------------------------------------------------------------------ #

    def track_value_in_region(
        self,
        source_region: Sequence[int],
        level: int,
        tracks: Sequence[TrackBelief],
    ) -> float:
        """Uncertainty of known tracks inside a region, normalised to [0, 1]."""
        x1, y1, x2, y2 = (float(v) for v in source_region)
        total = 0.0
        for track in tracks:
            if not (x1 <= track.center_x <= x2 and y1 <= track.center_y <= y2):
                continue
            if track.existence < self.min_track_existence:
                continue
            if track.best_zoom >= level:
                continue
            need = (level - track.best_zoom) / 2.0
            total += track.existence * (1.0 - track.confidence) * need
        return float(min(1.0, total))

    def value_of_information(
        self,
        level: int,
        center: Tuple[int, int],
        current_center: Tuple[int, int],
        max_delta: float,
        tracks: Sequence[TrackBelief],
        source_region: Optional[Sequence[int]] = None,
    ) -> float:
        """Score a candidate view by expected information gain minus travel."""
        if source_region is None:
            source_region = self._region_for(level, center)

        explore = self.unobserved_fraction(source_region, level)
        verify = self.track_value_in_region(source_region, level, tracks)

        distance = math.hypot(
            center[0] - current_center[0], center[1] - current_center[1]
        )
        travel = distance / max(float(max_delta), 1.0)

        return (
            self.explore_weight * explore
            + self.verify_weight * verify
            - self.travel_weight * travel
        )

    @staticmethod
    def _region_for(level: int, center: Tuple[int, int]) -> Tuple[int, int, int, int]:
        from utils import source_region_for_view

        return source_region_for_view(level, int(center[0]), int(center[1]))

    # ------------------------------------------------------------------ #
    # Candidate generation
    # ------------------------------------------------------------------ #

    def next_exploration_target(
        self,
        level: int,
        current_center: Tuple[int, int],
        center_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Nearest cell not yet observed at ``level``, clamped to legal centres.

        Ties break by row then column, so the order is deterministic. Returns
        ``None`` when the field is fully covered at this level.
        """
        mask = self.observed_level < level
        if not np.any(mask):
            return None

        rows, columns = np.nonzero(mask)
        candidate_x = (columns + 0.5) * self.cell_size
        candidate_y = (rows + 0.5) * self.cell_size
        distances = (candidate_x - current_center[0]) ** 2 + (
            candidate_y - current_center[1]
        ) ** 2
        order = np.lexsort((columns, rows, distances))
        best = int(order[0])
        target = (int(round(candidate_x[best])), int(round(candidate_y[best])))
        return self._clamp_target(target, center_bounds)

    def _clamp_target(
        self,
        target: Tuple[int, int],
        center_bounds: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int]:
        if center_bounds is None:
            return target
        minimum_x, maximum_x, minimum_y, maximum_y = center_bounds
        return (
            int(min(max(target[0], minimum_x), maximum_x)),
            int(min(max(target[1], minimum_y), maximum_y)),
        )

    def build_candidates(
        self,
        level: int,
        current_center: Tuple[int, int],
        tracks: Sequence[TrackBelief],
        max_delta: float,
        center_bounds: Optional[Tuple[int, int, int, int]] = None,
        include_exploration: bool = True,
        top_k: int = 8,
    ) -> List[CandidateView]:
        """Score verification and exploration candidates for one level."""
        candidates: List[CandidateView] = []

        for track in tracks:
            if track.existence < self.min_track_existence:
                continue
            if track.best_zoom >= level:
                continue
            center = self._clamp_target(
                (int(round(track.center_x)), int(round(track.center_y))),
                center_bounds,
            )
            value = self.value_of_information(
                level, center, current_center, max_delta, tracks
            )
            candidates.append(
                CandidateView(level, center[0], center[1], value, "verify")
            )

        if include_exploration:
            target = self.next_exploration_target(level, current_center, center_bounds)
            if target is not None:
                value = self.value_of_information(
                    level, target, current_center, max_delta, tracks
                )
                candidates.append(
                    CandidateView(level, target[0], target[1], value, "explore")
                )

        candidates.sort(key=lambda c: (-c.value, c.center_y, c.center_x))
        return candidates[:top_k]

    def best_candidate(
        self,
        level: int,
        current_center: Tuple[int, int],
        tracks: Sequence[TrackBelief],
        max_delta: float,
        center_bounds: Optional[Tuple[int, int, int, int]] = None,
        include_exploration: bool = True,
    ) -> Optional[CandidateView]:
        candidates = self.build_candidates(
            level,
            current_center,
            tracks,
            max_delta,
            center_bounds,
            include_exploration,
        )
        return candidates[0] if candidates else None
