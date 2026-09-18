"""Pure helpers of the exact-view dataset generator."""

import pytest

from dtos import OBJECT_CLASSES
from offline.build_exact_view_dataset import _annotation_to_yolo, grid_centers

FULL_FRAME = (0, 0, 3840, 2160)


def test_grid_centers_level0_is_the_full_frame_center():
    assert grid_centers(0) == [(1920, 1080)]


def test_grid_centers_level1_is_a_nine_cell_grid():
    centers = grid_centers(1)
    assert len(centers) == 9
    assert all(960 <= x <= 2880 and 540 <= y <= 1620 for x, y in centers)


def test_grid_centers_level2_is_a_seven_by_seven_grid():
    centers = grid_centers(2)
    assert len(centers) == 49
    assert all(480 <= x <= 3360 and 270 <= y <= 1890 for x, y in centers)


def test_annotation_to_yolo_normalizes_and_maps_class():
    annotation = {"object_id": "tank", "bbox": [0, 0, 384, 216]}
    row = _annotation_to_yolo(annotation, FULL_FRAME)

    assert row is not None
    assert row[0] == OBJECT_CLASSES.index("tank")
    assert row[1:] == pytest.approx((0.05, 0.05, 0.1, 0.1))


def test_annotation_to_yolo_drops_unknown_class():
    assert _annotation_to_yolo({"object_id": "car", "bbox": [0, 0, 10, 10]}, FULL_FRAME) is None


def test_annotation_to_yolo_drops_mostly_outside_crop():
    region = (1000, 1000, 1960, 1540)
    annotation = {"object_id": "tank", "bbox": [960, 960, 1010, 1010]}
    assert _annotation_to_yolo(annotation, region) is None
