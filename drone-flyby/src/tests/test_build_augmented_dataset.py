"""Tests for the copy-paste augmented exact-view dataset builder."""

import numpy as np
import pytest

from dtos import TRANSMITTED_VIEW_SIZE
from offline.build_augmented_dataset import (
    ObjectCrop,
    build_augmented_dataset,
    crop_objects_from_view,
    paste_augment,
)


VIEW_WIDTH, VIEW_HEIGHT = TRANSMITTED_VIEW_SIZE


def _blank_view() -> np.ndarray:
    return np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)


def test_crop_objects_from_view_extracts_each_label():
    view = _blank_view()
    view[240:300, 430:530] = 200
    rows = [(0, 0.5, 0.5, 0.1, 0.1)]

    crops = crop_objects_from_view(view, rows)

    assert len(crops) == 1
    assert crops[0].class_index == 0
    assert crops[0].image.shape[0] > 60  # includes margin
    assert crops[0].image.shape[1] > 100


def test_crop_objects_from_view_skips_tiny_boxes():
    view = _blank_view()
    rows = [(2, 0.5, 0.5, 0.0005, 0.0005)]

    assert crop_objects_from_view(view, rows, minimum_pixels=4) == []


def test_paste_augment_adds_a_label_and_pixels():
    view = _blank_view()
    view[240:300, 430:530] = 200
    rows = [(0, 0.5, 0.5, 0.1, 0.1)]
    bank = crop_objects_from_view(view, rows)

    augmented, new_rows = paste_augment(
        view, rows, bank, np.random.default_rng(0), max_pastes=1
    )

    assert len(new_rows) == len(rows) + 1
    assert np.any(augmented != view)
    class_index, cx, cy, width, height = new_rows[-1]
    assert class_index == 0
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0
    assert 0.0 < width < 1.0 and 0.0 < height < 1.0


def test_paste_augment_is_deterministic_for_a_seed():
    view = _blank_view()
    rows = [(0, 0.2, 0.2, 0.1, 0.1)]
    bank = [
        ObjectCrop(class_index=3, image=np.full((30, 40, 3), 120, np.uint8), width=40, height=30),
        ObjectCrop(class_index=5, image=np.full((24, 24, 3), 200, np.uint8), width=24, height=24),
    ]

    first = paste_augment(view, rows, bank, np.random.default_rng(7), max_pastes=2)
    second = paste_augment(view, rows, bank, np.random.default_rng(7), max_pastes=2)

    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_paste_augment_with_empty_bank_is_a_copy():
    view = _blank_view()
    rows = [(1, 0.4, 0.4, 0.1, 0.1)]

    augmented, new_rows = paste_augment(view, rows, [], np.random.default_rng(0))

    assert np.array_equal(augmented, view)
    assert new_rows == rows


def test_paste_augment_respects_zero_pastes():
    view = _blank_view()
    rows = [(1, 0.4, 0.4, 0.1, 0.1)]
    bank = [ObjectCrop(class_index=1, image=np.zeros((10, 10, 3), np.uint8), width=10, height=10)]

    augmented, new_rows = paste_augment(view, rows, bank, np.random.default_rng(0), max_pastes=0)

    assert len(new_rows) == 1
    assert np.array_equal(augmented, view)


def test_build_augmented_dataset_writes_a_usable_tree(tmp_path):
    data_yaml = build_augmented_dataset(
        scene="helsinki",
        output_dir=tmp_path,
        levels=[1],
        val_fraction=0.5,
        step_fraction=1.0,
        copy_paste_per_view=1,
        copies_per_source=2,
        frames=[0, 1],
        seed=3,
    )

    assert data_yaml.is_file()
    dataset_dir = data_yaml.parent
    train_images = list((dataset_dir / "images" / "train").glob("*.png"))
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_images = list((dataset_dir / "images" / "val").glob("*.png"))

    assert train_images and train_labels
    assert val_images
    assert "names:" in data_yaml.read_text()
    assert (dataset_dir / "augmentation_stats.json").is_file()
