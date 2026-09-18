"""Unit tests for the WorldMapTracker spatial memory system."""

import cv2
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

