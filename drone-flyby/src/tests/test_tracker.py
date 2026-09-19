"""Unit tests for the WorldMapTracker spatial memory system."""

import cv2
import numpy as np
import pytest
from core.interfaces import DetectionResult
from core.tracker import (
    TrackedObject,
    WorldMapTracker,
    apply_class_aware_nms,
    compute_iou,
)
from dtos import DroneFlybyPredictionDto
from utils import load_annotations, load_frame


SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160


def test_current_freshness_ranking_can_suppress_an_accurate_memory_box():
    """Characterize the current policy; do not change it without a score ablation."""
    truth = [0.4, 0.4, 0.5, 0.5]
    fresh_box = [0.436, 0.4, 0.536, 0.5]
    memory = DroneFlybyPredictionDto(object_id="tank", bbox=truth, confidence=0.49 * 0.9)
    fresh = DroneFlybyPredictionDto(object_id="tank", bbox=fresh_box, confidence=0.5 + 0.5 * 0.001)
    assert 0.45 < compute_iou(truth, fresh_box) < 0.50
    assert apply_class_aware_nms([memory, fresh], iou_threshold=0.45) == [fresh]


def _load_real_detections(frame_index: int):
    annotations = load_annotations(frame_index, scene="helsinki")
    return [
        DetectionResult(
            class_name=annotation["object_id"],
            bbox_global=(
                annotation["bbox"][0] / SOURCE_WIDTH,
                annotation["bbox"][1] / SOURCE_HEIGHT,
                annotation["bbox"][2] / SOURCE_WIDTH,
                annotation["bbox"][3] / SOURCE_HEIGHT,
            ),
            confidence=0.99,
            zoom_level=0,
            source_pixel_bbox=(
                float(annotation["bbox"][0]),
                float(annotation["bbox"][1]),
                float(annotation["bbox"][2]),
                float(annotation["bbox"][3]),
            ),
        )
        for annotation in annotations
    ]


def _load_real_gray(frame_index: int):
    image = load_frame(frame_index, scene="helsinki")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def test_track_confirmation_and_ego_motion():
    tracker = WorldMapTracker(
        min_hits_to_confirm=2,
        default_shift=(0.0, 50.0),
    )
    tracker.reset("test_seq")

    # Frame 0: First observation at (1000, 500, 1100, 600)
    det_f0 = [
        DetectionResult(
            class_name="tank",
            bbox_global=(1000 / 3840, 500 / 2160, 1100 / 3840, 600 / 2160),
            confidence=0.8,
            zoom_level=0,
            source_pixel_bbox=(1000.0, 500.0, 1100.0, 600.0),
        )
    ]
    preds_f0 = tracker.update(
        detections=det_f0,
        zoom_level=0,
        source_region_xyxy=(0, 0, 3840, 2160),
        frame_index=0,
    )
    # With min_hits_to_confirm=2 and confidence < 0.85, a single hit may not be confirmed yet
    assert len(tracker.tracks) == 1
    track = list(tracker.tracks.values())[0]
    assert track.hits == 1

    # Frame 1: Drone moves forward by 50px downward
    # Ground object now appears at y + 50 = (1000, 550, 1100, 650)
    det_f1 = [
        DetectionResult(
            class_name="tank",
            bbox_global=(1000 / 3840, 550 / 2160, 1100 / 3840, 650 / 2160),
            confidence=0.85,
            zoom_level=0,
            source_pixel_bbox=(1000.0, 550.0, 1100.0, 650.0),
        )
    ]
    preds_f1 = tracker.update(
        detections=det_f1,
        zoom_level=0,
        source_region_xyxy=(0, 0, 3840, 2160),
        frame_index=1,
    )

    # Track should now be confirmed and emitted!
    assert len(preds_f1) == 1
    assert preds_f1[0].object_id == "tank"
    assert track.hits == 2


def test_out_of_view_persistence():
    """Verify that objects are still reported when the camera moves elsewhere."""
    tracker = WorldMapTracker(
        min_hits_to_confirm=2,
        default_shift=(0.0, 50.0),
    )
    tracker.reset("test_seq")

    # Establish confirmed track in top-left region over frames 0 & 1
    for f_idx in [0, 1]:
        y_shift = f_idx * 50.0
        dets = [
            DetectionResult(
                class_name="helicopter",
                bbox_global=(500 / 3840, (500 + y_shift) / 2160, 600 / 3840, (600 + y_shift) / 2160),
                confidence=0.9,
                zoom_level=0,
                source_pixel_bbox=(500.0, 500.0 + y_shift, 600.0, 600.0 + y_shift),
            )
        ]
        tracker.update(
            detections=dets,
            zoom_level=0,
            source_region_xyxy=(0, 0, 3840, 2160),
            frame_index=f_idx,
        )

    # Frame 2: Camera zooms into bottom-right quadrant at Level 2
    # The helicopter (around x=500, y=650) is completely out of view!
    empty_detections_in_crop = []
    crop_source_region = (2800, 1200, 3760, 1740)  # Far bottom-right

    preds_f2 = tracker.update(
        detections=empty_detections_in_crop,
        zoom_level=2,
        source_region_xyxy=crop_source_region,
        frame_index=2,
    )

    # The tracker MUST still report the helicopter out-of-view!
    assert len(preds_f2) == 1
    assert preds_f2[0].object_id == "helicopter"
    # Its y-position should have advanced by another 50px: 550 + 50 = 600
    expected_y1_norm = 600.0 / 2160.0
    assert preds_f2[0].bbox[1] == pytest.approx(expected_y1_norm, abs=0.01)


def test_cross_resolution_refinement():
    tracker = WorldMapTracker(min_hits_to_confirm=1, default_shift=(0.0, 0.0))
    tracker.reset("test_seq")

    # Frame 0 at Level 0: coarse observation
    det_l0 = [
        DetectionResult(
            class_name="small_launcher",
            bbox_global=(0.2, 0.2, 0.25, 0.25),
            confidence=0.3,
            zoom_level=0,
            source_pixel_bbox=(768.0, 432.0, 960.0, 540.0),
        )
    ]
    tracker.update(detections=det_l0, zoom_level=0, source_region_xyxy=(0, 0, 3840, 2160), frame_index=0)
    track = list(tracker.tracks.values())[0]
    assert track.best_zoom == 0
    assert track.class_name == "small_launcher"

    # Frame 1 at Level 2: high-resolution observation refines class & bbox
    det_l2 = [
        DetectionResult(
            class_name="medium_launcher",
            bbox_global=(0.21, 0.21, 0.24, 0.24),
            confidence=0.85,
            zoom_level=2,
            source_pixel_bbox=(806.4, 453.6, 921.6, 518.4),
        )
    ]
    tracker.update(
        detections=det_l2,
        zoom_level=2,
        source_region_xyxy=(700, 350, 1660, 890),
        frame_index=1,
    )
    assert track.best_zoom == 2
    assert track.class_name == "medium_launcher"
    assert track.confidence == 0.85


def test_out_of_bounds_pruning():
    tracker = WorldMapTracker(default_shift=(0.0, 200.0))
    tracker.reset("test_seq")

    # Object near bottom edge y=2100
    det = [
        DetectionResult(
            class_name="tank",
            bbox_global=(0.5, 2100 / 2160, 0.6, 2150 / 2160),
            confidence=0.9,
            zoom_level=0,
            source_pixel_bbox=(1920.0, 2100.0, 2304.0, 2150.0),
        )
    ]
    tracker.update(detections=det, zoom_level=0, source_region_xyxy=(0, 0, 3840, 2160), frame_index=0)
    assert len(tracker.tracks) == 1

    # Next frame, shift by +200px moves it past y=2160 (out of bounds)
    tracker.update(detections=[], zoom_level=0, source_region_xyxy=(0, 0, 3840, 2160), frame_index=1)
    assert len(tracker.tracks) == 0


def test_class_aware_nms_suppresses_same_class_duplicates():
    preds = [
        DroneFlybyPredictionDto(object_id="tank", bbox=(0.1, 0.1, 0.3, 0.3), confidence=0.95),
        DroneFlybyPredictionDto(object_id="tank", bbox=(0.12, 0.12, 0.31, 0.31), confidence=0.90),
        DroneFlybyPredictionDto(object_id="helicopter", bbox=(0.11, 0.11, 0.29, 0.29), confidence=0.92),
    ]

    kept = apply_class_aware_nms(preds, iou_threshold=0.45)

    assert len(kept) == 2
    assert {pred.object_id for pred in kept} == {"tank", "helicopter"}


def test_real_helsinki_frame_pair_confirms_oracle_tracks():
    tracker = WorldMapTracker(
        min_hits_to_confirm=2,
        default_shift=(0.0, 58.0),
    )
    tracker.reset("helsinki_real_pair")

    detections_f0 = _load_real_detections(0)
    detections_f1 = _load_real_detections(1)

    tracker.update(
        detections=detections_f0,
        zoom_level=0,
        source_region_xyxy=(0, 0, SOURCE_WIDTH, SOURCE_HEIGHT),
        frame_index=0,
        l0_image_gray=_load_real_gray(0),
    )

    preds_f1 = tracker.update(
        detections=detections_f1,
        zoom_level=0,
        source_region_xyxy=(0, 0, SOURCE_WIDTH, SOURCE_HEIGHT),
        frame_index=1,
        l0_image_gray=_load_real_gray(1),
    )

    expected_ids = {det.class_name for det in detections_f0} & {det.class_name for det in detections_f1}
    predicted_ids = {pred.object_id for pred in preds_f1}

    assert expected_ids == {
        "condor",
        "helicopter",
        "jammer",
        "large_launcher",
        "mine_roller",
        "small_launcher",
        "small_plane",
        "small_tower",
        "spacecraft",
        "ta-ta",
        "tank",
    }
    # A class-aware centre-distance fallback keeps fast small objects such as
    # tank associated after a rigid shift, so the oracle is fully recovered.
    assert predicted_ids == expected_ids
    assert len(preds_f1) == len(expected_ids)


def test_class_aware_association_keeps_fast_small_object_attached():
    """A small objects outruns its own box after a rigid shift; centre-distance
    fallback keeps it on the same track instead of spawning a duplicate."""
    tracker = WorldMapTracker(min_hits_to_confirm=1, default_shift=(0.0, 0.0))
    tracker.reset("fast_small")

    tracker.update(
        detections=[
            DetectionResult(
                class_name="ta-ta",
                bbox_global=(1000 / 3840, 500 / 2160, 1032 / 3840, 517 / 2160),
                confidence=0.9,
                zoom_level=0,
                source_pixel_bbox=(1000.0, 500.0, 1032.0, 517.0),
            )
        ],
        zoom_level=0,
        source_region_xyxy=(0, 0, 3840, 2160),
        frame_index=0,
    )

    # 58 px of motion is larger than the object's 17 px height: IoU is 0.
    preds = tracker.update(
        detections=[
            DetectionResult(
                class_name="ta-ta",
                bbox_global=(1000 / 3840, 558 / 2160, 1032 / 3840, 575 / 2160),
                confidence=0.9,
                zoom_level=0,
                source_pixel_bbox=(1000.0, 558.0, 1032.0, 575.0),
            )
        ],
        zoom_level=0,
        source_region_xyxy=(0, 0, 3840, 2160),
        frame_index=1,
    )

    assert len(tracker.tracks) == 1
    track = list(tracker.tracks.values())[0]
    assert track.hits == 2
    assert len(preds) == 1


def test_in_view_miss_decays_faster_than_out_of_view_miss():
    tracker = WorldMapTracker(min_hits_to_confirm=1, default_shift=(0.0, 0.0))
    tracker.reset("negative_evidence")

    tracker.update(
        detections=[
            DetectionResult(
                class_name="tank",
                bbox_global=(100 / 3840, 100 / 2160, 200 / 3840, 200 / 2160),
                confidence=0.9,
                zoom_level=0,
                source_pixel_bbox=(100.0, 100.0, 200.0, 200.0),
            ),
            DetectionResult(
                class_name="helicopter",
                bbox_global=(2000 / 3840, 1000 / 2160, 2100 / 2160, 2160 / 3840),
                confidence=0.9,
                zoom_level=0,
                source_pixel_bbox=(2000.0, 1000.0, 2100.0, 1100.0),
            ),
        ],
        zoom_level=0,
        source_region_xyxy=(0, 0, 3840, 2160),
        frame_index=0,
    )

    # Crop contains the tank but not the helicopter, so only the tank was
    # actually observed to be absent.
    tracker.update(
        detections=[],
        zoom_level=2,
        source_region_xyxy=(50, 50, 1010, 590),
        frame_index=1,
    )

    tank = tracker.tracks[0]
    helicopter = tracker.tracks[1]
    assert tank.existence < helicopter.existence


def test_predict_only_returns_last_belief_without_new_observations():
    tracker = WorldMapTracker(min_hits_to_confirm=1, default_shift=(0.0, 0.0))
    tracker.reset("predict_only")

    emitted = tracker.update(
        detections=[
            DetectionResult(
                class_name="jammer",
                bbox_global=(0.5, 0.5, 0.52, 0.52),
                confidence=0.9,
                zoom_level=0,
                source_pixel_bbox=(1920.0, 1080.0, 1996.8, 1123.2),
            )
        ],
        zoom_level=0,
        source_region_xyxy=(0, 0, 3840, 2160),
        frame_index=0,
    )
    predicted = tracker.predict_only()

    assert [p.object_id for p in predicted] == [p.object_id for p in emitted]


def test_real_helsinki_tracks_persist_when_current_view_is_empty():
    tracker = WorldMapTracker(
        min_hits_to_confirm=2,
        default_shift=(0.0, 58.0),
    )
    tracker.reset("helsinki_real_persistence")

    tracker.update(
        detections=_load_real_detections(0),
        zoom_level=0,
        source_region_xyxy=(0, 0, SOURCE_WIDTH, SOURCE_HEIGHT),
        frame_index=0,
        l0_image_gray=_load_real_gray(0),
    )
    tracker.update(
        detections=_load_real_detections(1),
        zoom_level=0,
        source_region_xyxy=(0, 0, SOURCE_WIDTH, SOURCE_HEIGHT),
        frame_index=1,
        l0_image_gray=_load_real_gray(1),
    )

    preds_empty_view = tracker.update(
        detections=[],
        zoom_level=2,
        source_region_xyxy=(2800, 1200, 3760, 1740),
        frame_index=2,
        l0_image_gray=_load_real_gray(2),
    )

    predicted_ids = {pred.object_id for pred in preds_empty_view}

    assert "helicopter" in predicted_ids
    assert "jammer" in predicted_ids
    assert "large_launcher" in predicted_ids
    assert len(preds_empty_view) >= 8


def test_summary_lists_only_confirmed_tracks():
    tracker = WorldMapTracker(
        min_hits_to_confirm=2,
        single_hit_confirm_confidence=0.5,
        default_shift=(0.0, 0.0),
    )
    tracker.reset("summary_confirmed")

    weak_detection = [
        DetectionResult(
            class_name="tank",
            bbox_global=(0.1, 0.1, 0.2, 0.2),
            confidence=0.1,
            zoom_level=0,
            source_pixel_bbox=(384.0, 216.0, 768.0, 432.0),
        )
    ]

    tracker.update(weak_detection, 0, (0, 0, 3840, 2160), frame_index=0)
    assert tracker.get_summary().unscanned_clusters == []

    tracker.update(weak_detection, 0, (0, 0, 3840, 2160), frame_index=1)
    assert len(tracker.get_summary().unscanned_clusters) == 1


def test_summary_exports_track_beliefs_for_confirmed_tracks():
    tracker = WorldMapTracker(min_hits_to_confirm=1, default_shift=(0.0, 0.0))
    tracker.reset("belief_export")

    tracker.update(
        detections=[
            DetectionResult(
                class_name="jammer",
                bbox_global=(0.5, 0.5, 0.52, 0.52),
                confidence=0.8,
                zoom_level=1,
                source_pixel_bbox=(1920.0, 1080.0, 1996.8, 1123.2),
            )
        ],
        zoom_level=1,
        source_region_xyxy=(960, 540, 2880, 1620),
        frame_index=0,
    )

    summary = tracker.get_summary()
    assert len(summary.track_beliefs) == 1
    belief = summary.track_beliefs[0]
    assert belief.class_name == "jammer"
    assert belief.center_x == pytest.approx(1958.4, abs=1.0)
    assert belief.center_y == pytest.approx(1101.6, abs=1.0)
    assert belief.best_zoom == 1
    assert 0.0 <= belief.existence <= 1.0


def test_camera_delta_corrects_measured_ego_motion():
    """When the camera itself moved, the measured content shift understates the
    true ego-motion; the known centre delta must be added back."""
    base = cv2.cvtColor(load_frame(0, scene="helsinki"), cv2.COLOR_BGR2GRAY)
    # Content moves down 20 px between frames. The camera also moves down 10 px
    # (so the cr(ab appears to move only 10 px).
    shifted = cv2.warpAffine(base, np.float32([[1, 0, 0], [0, 1, 20]]), (base.shape[1], base.shape[0]))

    region_f0 = (960, 540, 2880, 1620)
    region_f1 = (960, 550, 2880, 1630)  # centre moved down 10
    gray_f0 = base[region_f0[1]:region_f0[3], region_f0[0]:region_f0[2]]
    gray_f1 = shifted[region_f1[1]:region_f1[3], region_f1[0]:region_f1[2]]

    tracker = WorldMapTracker(default_shift=(0.0, 0.0))
    tracker.reset("ego_correction")

    tracker._estimate_ego_motion(gray_f0, 0, region_f0)
    shift = tracker._estimate_ego_motion(gray_f1, 1, region_f1)

    # 0.7 * (measured 10 + camera delta 10) + 0.3 * prior 0
    assert shift[1] == pytest.approx(14.0, abs=2.0)


def test_ego_motion_registers_overlap_across_a_level_change():
    base = cv2.cvtColor(load_frame(0, scene="helsinki"), cv2.COLOR_BGR2GRAY)
    region_l1 = (960, 540, 2880, 1620)
    region_l2 = (1440, 810, 2400, 1350)
    gray_l1 = base[region_l1[1]:region_l1[3], region_l1[0]:region_l1[2]]
    gray_l2 = base[region_l2[1]:region_l2[3], region_l2[0]:region_l2[2]]

    tracker = WorldMapTracker(default_shift=(0.0, 58.0))
    tracker.reset("ego_scale")

    tracker._estimate_ego_motion(gray_l1, 0, region_l1)
    shift = tracker._estimate_ego_motion(gray_l2, 1, region_l2)

    # Both views show stationary terrain; only the prior's smoothing term remains.
    assert shift[0] == pytest.approx(0.0, abs=2.0)
    assert shift[1] == pytest.approx(17.4, abs=2.0)


def _small_detection(box, level=2, name="tank"):
    x1, y1, x2, y2 = box
    return DetectionResult(name, (x1 / 3840, y1 / 2160, x2 / 3840, y2 / 2160),
                           0.9, level, tuple(float(v) for v in box))


def test_lower_zoom_reanchors_position_without_destroying_precise_size():
    tracker = WorldMapTracker(default_shift=(0, 0), min_hits_to_confirm=1)
    tracker.update([_small_detection((100, 100, 140, 140))], 2, (0, 0, 960, 540), 0)
    tracker.update([_small_detection((130, 90, 190, 150), 1)], 1, (0, 0, 1920, 1080), 1)
    assert tracker.tracks[0].bbox_4k == pytest.approx((140, 100, 180, 140))
    assert tracker.tracks[0].best_zoom == 2


def test_recent_reobservation_resets_expiry_not_lifetime_miss_count():
    tracker = WorldMapTracker(default_shift=(0, 0), min_hits_to_confirm=1, out_of_view_max_age=2)
    detection = _small_detection((100, 100, 140, 140))
    for frame in range(12):
        if frame % 2 == 0:
            tracker.update([detection], 2, (0, 0, 960, 540), frame)
        else:
            tracker.update([], 2, (2000, 1000, 2960, 1540), frame)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].hits == 6


def test_predict_only_advances_without_negative_observation():
    tracker = WorldMapTracker(default_shift=(0, 20), min_hits_to_confirm=1)
    tracker.update([_small_detection((100, 100, 140, 140))], 2, (0, 0, 960, 540), 0)
    predicted = tracker.predict_only(frame_index=3)
    assert predicted[0].bbox[1] * 2160 == pytest.approx(160)
    assert tracker.tracks[0].frames_since_seen == 3
    assert tracker.tracks[0].existence > 0.7
    tracker.predict_only(frame_index=3)
    assert tracker.tracks[0].bbox_4k[1] == pytest.approx(160)


def test_assignment_prefers_a_valid_distance_match_over_invalid_tie():
    tracker = WorldMapTracker(default_shift=(0, 0), min_hits_to_confirm=1)
    tracker.update([
        _small_detection((100, 100, 140, 140)),
        _small_detection((1000, 500, 1030, 530), name="ta-ta"),
    ], 0, (0, 0, 3840, 2160), 0)
    tracker.update([_small_detection((1060, 500, 1090, 530), name="ta-ta")],
                   0, (0, 0, 3840, 2160), 1)
    assert len(tracker.tracks) == 2
    assert tracker.tracks[1].hits == 2


def test_tiny_in_view_miss_is_not_strong_negative_evidence():
    tracker = WorldMapTracker(default_shift=(0, 0), min_hits_to_confirm=1)
    tracker.update([_small_detection((100, 100, 120, 120))], 2, (0, 0, 960, 540), 0)
    tracker.update([], 0, (0, 0, 3840, 2160), 1)
    assert tracker.tracks[0].existence > 0.7


def test_uncertainty_does_not_grow_quadratically_with_frame_gap():
    trackers = [WorldMapTracker(default_shift=(0, 20), min_hits_to_confirm=1) for _ in range(2)]
    for tracker in trackers:
        tracker.update([_small_detection((100, 100, 140, 140))], 2, (0, 0, 960, 540), 0)
    trackers[0].predict_only(frame_index=1)
    trackers[1].predict_only(frame_index=4)
    assert trackers[1].tracks[0].position_std <= 4 * trackers[0].tracks[0].position_std


def test_crop_edge_observation_anchors_visible_edge_without_shrinking_track():
    tracker = WorldMapTracker(default_shift=(0, 0), min_hits_to_confirm=1)
    tracker.update([_small_detection((940, 400, 1040, 500))], 0, (0, 0, 3840, 2160), 0)
    tracker.update([_small_detection((960, 400, 1040, 500))], 2, (960, 270, 1920, 810), 1)
    assert tracker.tracks[0].bbox_4k == pytest.approx((940, 400, 1040, 500))
    assert tracker.tracks[0].best_zoom == 0


def test_fresh_weak_detection_is_emitted_without_premature_memory_confirmation():
    tracker = WorldMapTracker(default_shift=(0, 0))
    detection = _small_detection((100, 100, 140, 140))
    detection.confidence = 0.001
    output = tracker.update([detection], 2, (0, 0, 960, 540), 0)
    assert len(output) == 1
    assert output[0].confidence == pytest.approx(0.5005)
    assert not tracker._is_confirmed(tracker.tracks[0])


def test_speculative_memory_does_not_degrade_fresh_detection_ap():
    from local_evaluator import score_ground_truth

    tracker = WorldMapTracker(default_shift=(0, 0))
    false_positive = _small_detection((3000, 1000, 3040, 1040))
    true_positive = _small_detection((100, 100, 140, 140))
    true_positive.confidence = 0.001
    first = tracker.update([false_positive], 0, (0, 0, 3840, 2160), 0)
    second = tracker.update([true_positive], 2, (0, 0, 960, 540), 1)
    assert second[0].bbox[0] * 3840 == pytest.approx(100)
    assert second[0].confidence > second[1].confidence
    truth = {0: [], 1: [{"object_id": "tank", "bbox": [100, 100, 140, 140]}]}
    def prediction(box, confidence):
        return {"object_id": "tank", "bbox": box, "confidence": confidence}
    baseline = {
        0: [prediction(false_positive.source_pixel_bbox, false_positive.confidence)],
        1: [prediction(true_positive.source_pixel_bbox, true_positive.confidence)],
    }
    with_memory = {
        frame: [prediction([p.bbox[0] * 3840, p.bbox[1] * 2160,
                            p.bbox[2] * 3840, p.bbox[3] * 2160], p.confidence) for p in predictions]
        for frame, predictions in enumerate([first, second])
    }
    assert score_ground_truth(truth, with_memory)[0] >= score_ground_truth(truth, baseline)[0]


def test_vectorized_association_preserves_scalar_gates_costs_and_assignments(monkeypatch):
    from scipy.optimize import linear_sum_assignment
    from core.tracker import TrackedObject, compute_iou
    import core.tracker as tracker_module

    captured = []
    def capture_cost(cost):
        captured.append(cost.copy())
        return linear_sum_assignment(cost)
    monkeypatch.setattr(tracker_module, "linear_sum_assignment", capture_cost)

    rng = np.random.default_rng(91)
    classes = ["tank", "jammer", "helicopter"]
    for trial in range(40):
        tracker = WorldMapTracker()
        tracks = 1 + trial % 17
        count = 1 + trial % 13
        for index in range(tracks):
            x, y = rng.uniform(0, 500, 2)
            width, height = rng.uniform(5, 180, 2)
            box = tuple(float(v) for v in (x, y, x + width, y + height))
            tracker.tracks[index] = TrackedObject(index, classes[index % 3], box, 0.9, 1)
        detections = []
        for index in range(count):
            box = np.asarray(tracker.tracks[index % tracks].bbox_4k)
            dx, dy = rng.uniform(-180, 180, 2)
            box += (dx, dy, dx, dy)
            detections.append(_small_detection(box, name=classes[(index + trial) % 3]))
        ids = list(reversed(tracker.tracks))
        costs = np.full((count, tracks + count), 1e6, dtype=np.float32)
        costs[:, tracks:] = 1.5
        gates = np.zeros((count, tracks), dtype=bool)
        for row, detection in enumerate(detections):
            a = detection.source_pixel_bbox
            for column, tid in enumerate(ids):
                track = tracker.tracks[tid]
                b = track.bbox_4k
                overlap = compute_iou(a, b)
                distance = (((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) ** 2
                            + ((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) ** 2) ** 0.5
                diagonal = max(((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5, 1.0)
                same_class = detection.class_name == track.class_name
                if overlap >= tracker.iou_match_threshold or (same_class and distance <= max(diagonal, 150)):
                    gates[row, column] = True
                    costs[row, column] = (1 - overlap - 0.25 * max(0, 1 - distance / diagonal)
                                          + (0 if same_class else 0.2))
        rows, columns = linear_sum_assignment(costs)
        expected = [(int(row), ids[column]) for row, column in zip(rows, columns)
                    if column < tracks and gates[row, column]]
        actual, detection_ids, matched_tracks = tracker._associate(detections, ids)
        np.testing.assert_array_equal(captured[-1], costs)
        assert actual == expected
        assert detection_ids == {row for row, _ in expected}
        assert matched_tracks == {tid for _, tid in expected}


def test_vectorized_nms_matches_scalar_order_thresholds_and_global_cap():
    from collections import defaultdict
    from core.tracker import apply_class_aware_nms, compute_iou
    from dtos import DroneFlybyPredictionDto

    rng = np.random.default_rng(32)
    for threshold in (0.0, 0.2, 0.45, 0.7, 1.0):
        predictions = []
        for index in range(180):
            x, y = rng.uniform(0, 0.7, 2)
            width, height = rng.uniform(0.02, 0.3, 2)
            predictions.append(DroneFlybyPredictionDto(
                object_id=("tank", "jammer", "helicopter")[index % 3],
                bbox=(float(x), float(y), float(x + width), float(y + height)),
                confidence=float(rng.choice([0.1, 0.3, 0.6, 0.9])),
            ))
        grouped = defaultdict(list)
        for prediction in predictions:
            grouped[prediction.object_id].append(prediction)
        expected = []
        for group in grouped.values():
            kept = []
            for prediction in sorted(group, key=lambda item: -item.confidence):
                if not any(compute_iou(prediction.bbox, box) > threshold for box in kept):
                    expected.append(prediction)
                    kept.append(prediction.bbox)
        expected.sort(key=lambda item: -item.confidence)
        assert apply_class_aware_nms(predictions, threshold, max_total=37) == expected[:37]
