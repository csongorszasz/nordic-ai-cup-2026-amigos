"""Tests for the offline closed-loop camera-policy simulator."""

from typing import Dict, List

import numpy as np
import pytest

from config import DroneFlybyConfig
from dtos import IMAGE_HEIGHT, IMAGE_WIDTH
from offline.camera_simulator import (
    SimulatorReport,
    run_simulation,
    view_oracle_detections,
)


def _synthetic_frame() -> np.ndarray:
    frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    # A little texture so ego-motion has something to correlate.
    frame[::37, ::53] = 90
    return frame


def _annotations() -> List[Dict]:
    return [
        {"object_id": "tank", "bbox": [1870, 1030, 1970, 1130]},
        {"object_id": "hangar", "bbox": [220, 200, 420, 360]},
        {"object_id": "helicopter", "bbox": [3400, 1700, 3500, 1790]},
        {"object_id": "jammer", "bbox": [600, 1600, 660, 1660]},
    ]


def _frame_provider(frame: int) -> np.ndarray:
    return _synthetic_frame()


def _annotation_provider(frame: int) -> List[Dict]:
    return _annotations()


def _stub_score(scene, predictions):
    return 0.42, {"tank": 0.42}


def test_view_oracle_detections_filters_by_centre():
    annotations = _annotations()
    # A view around the tank only.
    detections = view_oracle_detections(annotations, (1440, 810, 2400, 1350), 2)

    assert [d.class_name for d in detections] == ["tank"]
    assert detections[0].zoom_level == 2
    assert detections[0].source_pixel_bbox == (1870.0, 1030.0, 1970.0, 1130.0)


def test_simulation_runs_and_stays_legal():
    config = DroneFlybyConfig(POLICY_TYPE="belief_voi", TRACKER_TYPE="world_map")
    report = run_simulation(
        config,
        scene="helsinki",
        frames=[0, 1, 2, 3, 4, 5],
        frame_provider=_frame_provider,
        annotation_provider=_annotation_provider,
        score_fn=_stub_score,
    )

    assert isinstance(report, SimulatorReport)
    assert report.frames == 6
    assert report.illegal_moves == 0
    assert report.map50 == pytest.approx(0.42)
    assert report.total_instances == 4
    assert report.coverage_l1 > 0.0


def test_belief_policy_observes_ground_at_l1():
    config = DroneFlybyConfig(POLICY_TYPE="belief_voi", TRACKER_TYPE="world_map")
    report = run_simulation(
        config,
        scene="helsinki",
        frames=list(range(8)),
        frame_provider=_frame_provider,
        annotation_provider=_annotation_provider,
        score_fn=_stub_score,
    )

    assert report.instances_seen_l1 >= 1


def test_hold_policy_never_reaches_l1():
    config = DroneFlybyConfig(POLICY_TYPE="hold", TRACKER_TYPE="passthrough")
    report = run_simulation(
        config,
        scene="helsinki",
        frames=list(range(8)),
        frame_provider=_frame_provider,
        annotation_provider=_annotation_provider,
        score_fn=_stub_score,
    )

    assert report.instances_seen_l1 == 0
    assert report.coverage_l1 == 0.0
    assert report.illegal_moves == 0


def test_summary_mentions_the_key_numbers():
    report = SimulatorReport(
        policy="belief_voi",
        tracker="world_map",
        frames=10,
        illegal_moves=0,
        coverage_l1=0.5,
        coverage_l2=0.1,
        instances_seen_l1=3,
        instances_seen_l2=1,
        total_instances=4,
        map50=0.75,
    )
    text = report.summary()
    assert "belief_voi" in text
    assert "0.7500" in text
