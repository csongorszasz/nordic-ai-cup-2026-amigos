"""Unit tests for the belief field used by information-gain camera planning."""

import pytest

from core.belief import BeliefField, CandidateView
from core.interfaces import TrackBelief
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH


FULL_FRAME = (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)


def _track(cx, cy, existence=0.9, confidence=0.3, best_zoom=0):
    return TrackBelief(
        class_name="tank",
        center_x=cx,
        center_y=cy,
        existence=existence,
        confidence=confidence,
        best_zoom=best_zoom,
        position_std=0.0,
    )


def test_grid_dimensions_and_initial_coverage():
    field = BeliefField(cell_size=120)

    assert field.columns == 32
    assert field.rows == 18
    assert field.coverage_fraction(0) == 0.0
    assert field.coverage_fraction(1) == 0.0
    assert not field.is_fully_covered(1)


def test_mark_observed_updates_coverage_and_deepest_level():
    field = BeliefField(cell_size=120)

    field.mark_observed((0, 0, 1920, 1080), level=1, frame_index=0)
    assert field.coverage_fraction(0) == pytest.approx(0.25)
    assert field.coverage_fraction(1) == pytest.approx(0.25)
    assert field.coverage_fraction(2) == 0.0

    # A later L2 view upgrades the cells it covers without lowering others.
    # The grid rounds outwards, so an L2 rectangle marks 8x5 cells rather than
    # exactly a sixteenth of the grid.
    field.mark_observed((0, 0, 960, 540), level=2, frame_index=1)
    assert field.coverage_fraction(2) == pytest.approx(1.0 / 16.0, abs=0.01)


def test_mark_observed_keeps_deepest_level():
    field = BeliefField(cell_size=120)

    field.mark_observed((0, 0, 960, 540), level=2, frame_index=0)
    field.mark_observed((0, 0, 960, 540), level=0, frame_index=1)

    window = field.observed_level[0:5, 0:8]
    assert window.min() == 2


def test_unobserved_fraction_of_a_region():
    field = BeliefField(cell_size=120)
    region = (0, 0, 1920, 1080)

    assert field.unobserved_fraction(region, 1) == pytest.approx(1.0)
    field.mark_observed(region, level=1, frame_index=0)
    assert field.unobserved_fraction(region, 1) == pytest.approx(0.0)
    assert field.unobserved_fraction(region, 2) == pytest.approx(1.0)


def test_next_exploration_target_is_nearest_unobserved_and_clamped():
    field = BeliefField(cell_size=120)

    target = field.next_exploration_target(1, (1920, 1080))
    assert target is not None
    # The nearest unobserved cell to the frame centre is around (1860, 1080)
    # after clamping into the legal L1 centre window.
    x, y = target
    assert 960 <= x <= 2880
    assert 540 <= y <= 1620


def test_next_exploration_target_returns_none_when_covered():
    field = BeliefField(cell_size=120)
    field.mark_observed(FULL_FRAME, level=1, frame_index=0)

    assert field.next_exploration_target(1, (1920, 1080)) is None


def test_exploration_targets_are_deterministic():
    field_a = BeliefField(cell_size=120)
    field_b = BeliefField(cell_size=120)

    assert field_a.next_exploration_target(1, (900, 600)) == field_b.next_exploration_target(
        1, (900, 600)
    )


def test_value_of_information_prefers_a_verification_over_the_same_explore_cell():
    field = BeliefField(cell_size=120)
    uncertain = [_track(2000, 1000, existence=1.0, confidence=0.1, best_zoom=0)]

    value_verify = field.value_of_information(
        2, (2000, 1000), (2000, 1000), 551.0, uncertain
    )
    value_explore_only = field.value_of_information(
        2, (2000, 1000), (2000, 1000), 551.0, []
    )

    assert value_verify > value_explore_only


def test_value_of_information_penalises_travel():
    field = BeliefField(cell_size=120)

    near = field.value_of_information(1, (1920, 1080), (1920, 1080), 1102.0, [])
    far = field.value_of_information(1, (3000, 1600), (500, 500), 1102.0, [])

    # Both are fully unobserved, so the only difference is travel distance.
    assert near > far


def test_build_candidates_includes_verify_and_explore():
    field = BeliefField(cell_size=120)
    tracks = [_track(2000, 1000)]

    candidates = field.build_candidates(
        1, (1920, 1080), tracks, 1102.0, (960, 2880, 540, 1620)
    )

    kinds = {candidate.kind for candidate in candidates}
    assert "verify" in kinds
    assert "explore" in kinds
    assert all(isinstance(candidate, CandidateView) for candidate in candidates)
    # Sorted by value descending.
    values = [candidate.value for candidate in candidates]
    assert values == sorted(values, reverse=True)


def test_build_candidates_skips_certain_tracks_already_seen_at_that_level():
    field = BeliefField(cell_size=120)
    tracks = [_track(2000, 1000, best_zoom=2, confidence=0.95)]

    candidates = field.build_candidates(
        2, (1920, 1080), tracks, 551.0, (480, 3360, 270, 1890),
        include_exploration=False,
    )

    assert candidates == []


def test_reset_clears_the_field():
    field = BeliefField(cell_size=120)
    field.mark_observed(FULL_FRAME, level=2, frame_index=0)
    assert field.is_fully_covered(2)

    field.reset()
    assert not field.is_fully_covered(0)


def test_new_terrain_is_not_permanently_marked_as_covered():
    field = BeliefField(cell_size=120)
    field.advance_to(0, (0, 58))
    field.mark_observed(FULL_FRAME, 2, 0)
    field.advance_to(1, (0, 58))
    assert not field.is_fully_covered(1)
    assert (field.observed_level[0] == -1).all()
    field.advance_to(3, (0, 58))
    assert field.coverage_fraction(1) < 1.0


def test_zoom_specific_observations_expire_independently():
    field = BeliefField(max_age_frames=3)
    field.advance_to(0, (0, 0))
    field.mark_observed(FULL_FRAME, 2, 0)
    field.advance_to(2, (0, 0))
    field.mark_observed(FULL_FRAME, 0, 2)
    field.advance_to(4, (0, 0))
    assert field.coverage_fraction(0) == 1.0
    assert field.coverage_fraction(1) == 0.0


def test_high_zoom_tracks_can_be_revisited_when_position_drifts():
    field = BeliefField()
    track = _track(2000, 1000, best_zoom=2, confidence=0.95)
    track.position_std = 30
    candidates = field.build_candidates(
        2, (1920, 1080), [track], 551.0, (480, 3360, 270, 1890),
        include_exploration=False,
    )
    assert len(candidates) == 1
