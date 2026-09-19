from core.camera_policy import AdaptiveCameraPolicy
from core.interfaces import TrackerSummary
from tests.test_camera_policy import SimulatedCamera, _build_request
import pytest


def summary(confident=True):
    return TrackerSummary(
        num_active_tracks=0, unscanned_clusters=[], current_shift_estimate=(0, 58),
        fresh_class_confidences={name: 0.8 for name in ("tank", "hangar", "jammer", "helicopter")}
        if confident else {"tank": 0.01},
    )


def test_reliable_full_views_stay_wide():
    policy = AdaptiveCameraPolicy()
    policy.reset("wide")
    camera = SimulatedCamera()
    for frame in range(30):
        assert policy.decide_next_view(_build_request(camera, frame, "wide"), summary()) is None


def test_weak_full_views_enter_exploration_without_detection_candidates():
    policy = AdaptiveCameraPolicy()
    policy.reset("weak")
    camera = SimulatedCamera()
    assert policy.decide_next_view(_build_request(camera, 0, "weak"), summary(False)) is None
    move = policy.decide_next_view(_build_request(camera, 1, "weak"), summary(False))
    assert move.resolution_level == 1
    camera.apply(move)


def test_audit_returns_legally_from_l2_and_reliable_view_restores_hold():
    policy = AdaptiveCameraPolicy(survey_interval=4)
    policy.reset("audit")
    camera = SimulatedCamera(resolution_level=2, center_x=3360, center_y=1890)
    first = policy.decide_next_view(_build_request(camera, 4, "audit"), summary(False))
    assert first.resolution_level == 1
    camera.apply(first)
    second = policy.decide_next_view(_build_request(camera, 5, "audit"), summary(False))
    assert second.resolution_level == 0
    camera.apply(second)
    assert policy.decide_next_view(_build_request(camera, 6, "audit"), summary()) is None


def test_old_memory_is_not_mistaken_for_fresh_full_view_evidence():
    policy = AdaptiveCameraPolicy()
    policy.reset("memory")
    camera = SimulatedCamera()
    stale = summary(False)
    stale.num_active_tracks = 100
    policy.decide_next_view(_build_request(camera, 0, "memory"), stale)
    assert policy.decide_next_view(_build_request(camera, 1, "memory"), stale).resolution_level == 1


@pytest.mark.parametrize("arguments", [
    {"min_confident_classes": 0}, {"survey_interval": 0}, {"confidence": float("nan")},
])
def test_invalid_quality_gate_fails_explicitly(arguments):
    with pytest.raises(ValueError):
        AdaptiveCameraPolicy(**arguments)
