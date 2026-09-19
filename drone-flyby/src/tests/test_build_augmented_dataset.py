"""Tests for the copy-paste augmented exact-view dataset builder."""

import numpy as np
import pytest
import json

from dtos import OBJECT_CLASSES, TRANSMITTED_VIEW_SIZE
import offline.build_augmented_dataset as builder
from offline.build_augmented_dataset import (
    ObjectCrop,
    build_augmented_dataset,
    crop_objects_from_view,
    paste_augment,
    rotate_object_crop,
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
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    assert {item["source_frame"] for item in manifest["bank"]} == {0}
    assert {item["frame"] for item in manifest["samples"] if item["split"] == "train"} == {0}
    assert {item["frame"] for item in manifest["samples"] if item["split"] == "val"} == {1}


def test_copy_paste_label_keeps_tight_box_not_context():
    rows = [(0, 0.5, 0.5, 0.1, 0.1)]
    bank = crop_objects_from_view(_blank_view(), rows, margin=0.5)
    _, labels = paste_augment(
        _blank_view(), [], bank, np.random.default_rng(1),
        scale_range=(1.0, 1.0), max_pastes=1,
    )
    assert labels[0][3:] == pytest.approx((0.1, 0.1))


def test_copy_paste_does_not_mix_zoom_scales():
    bank = crop_objects_from_view(_blank_view(), [(0, 0.5, 0.5, 0.1, 0.1)], zoom_level=2)
    _, labels = paste_augment(_blank_view(), [], bank, np.random.default_rng(1), zoom_level=0)
    assert labels == []


def test_augmented_dataset_retains_empty_views(tmp_path):
    data_yaml = build_augmented_dataset(
        scene="helsinki", output_dir=tmp_path, levels=[2],
        frames=[0, 1], step_fraction=1.0, neg_stride=0,
    )
    labels = list((data_yaml.parent / "labels").rglob("*.txt"))
    assert any(not path.read_text().strip() for path in labels)


def test_unaugmented_control_has_identical_views_labels_and_negative_sampling(tmp_path, monkeypatch):
    import offline.build_exact_view_dataset as exact

    monkeypatch.setattr(exact, "frame_numbers", lambda scene: [0, 1])
    control = exact.build_exact_view_dataset(
        "helsinki", tmp_path / "control", levels=[2], val_fraction=0.5,
        neg_stride=2, step_fraction=1.0,
    ).parent
    candidate = build_augmented_dataset(
        "helsinki", tmp_path / "candidate", levels=[2], val_fraction=0.5,
        neg_stride=2, step_fraction=1.0, frames=[0, 1],
    ).parent
    for folder, suffix in (("images", ".png"), ("labels", ".txt")):
        expected = sorted(path.relative_to(control) for path in (control / folder).rglob(f"*{suffix}"))
        actual = sorted(path.relative_to(candidate) for path in (candidate / folder).rglob(f"*{suffix}"))
        assert actual == expected
        for path in expected:
            assert (candidate / path).read_bytes() == (control / path).read_bytes()


@pytest.mark.parametrize("turns", range(4))
def test_quarter_turn_labels_match_the_rotated_foreground_pixels(turns):
    image = np.zeros((6, 10, 3), dtype=np.uint8)
    image[1:4, 2:7] = 255
    crop = ObjectCrop(3, image, 10, 6, object_box=(2, 1, 7, 4), zoom_level=2, source_frame=12, view_scale=0.25)
    rotated = rotate_object_crop(crop, turns)
    ys, xs = np.nonzero(rotated.image[:, :, 0])
    assert rotated.object_box == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    assert rotated.image.shape[:2] == (rotated.height, rotated.width)
    assert (rotated.class_index, rotated.zoom_level, rotated.source_frame) == (3, 2, 12)
    assert rotated.view_scale == 0.25


def test_fractional_tiny_box_is_fully_contained_in_its_crop():
    rows = [(0, (503.99 + 1.6) / VIEW_WIDTH, 0.5, 3.2 / VIEW_WIDTH, 6 / VIEW_HEIGHT)]
    crop = crop_objects_from_view(_blank_view(), rows)[0]
    x1, y1, x2, y2 = crop.object_box
    assert 0 <= x1 < x2 <= crop.width
    assert 0 <= y1 < y2 <= crop.height


def test_reviewed_source_pixels_follow_optical_scale_without_context_leak():
    rgba = np.full((20, 40, 4), (0, 255, 0, 0), np.uint8)
    rgba[5:15, 10:30] = (0, 0, 200, 255)
    alpha_crop = ObjectCrop(0, rgba, 40, 20, object_box=(0, 0, 40, 20),
                            zoom_level=0, view_scale=0.25)
    context = rgba.copy()
    context[:, :, 3] = 255
    context_crop = ObjectCrop(0, context, 40, 20, object_box=(0, 0, 40, 20),
                              zoom_level=0, view_scale=0.25)
    alpha_image, alpha_labels = paste_augment(
        _blank_view(), [], [alpha_crop], np.random.default_rng(1), scale_range=(1, 1), zoom_level=0,
    )
    context_image, context_labels = paste_augment(
        _blank_view(), [], [context_crop], np.random.default_rng(1), scale_range=(1, 1), zoom_level=0,
    )
    assert alpha_labels == context_labels
    assert alpha_labels[0][3:] == pytest.approx((10 / VIEW_WIDTH, 5 / VIEW_HEIGHT))
    assert not alpha_image[:, :, 1].any()
    assert context_image[:, :, 1].any()


def _reviewed_fixture():
    image = np.full((20, 32, 4), (0, 255, 0, 0), np.uint8)
    image[5:15, 8:24] = (0, 0, 200, 255)
    return ({name: image.copy() for name in OBJECT_CLASSES},
            {name: {"scene": "helsinki", "frame": 0} for name in OBJECT_CLASSES})


def test_reviewed_bank_cannot_import_validation_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(builder, "load_reviewed_assets", lambda *args: _reviewed_fixture())
    with pytest.raises(ValueError, match="validation source frame"):
        builder.reviewed_crop_bank("helsinki", {0: "val"}, [0, 1, 2], tmp_path, tmp_path, "alpha")


def test_context_and_alpha_training_controls_have_identical_labels_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "load_reviewed_assets", lambda *args: _reviewed_fixture())
    roots = []
    for mode in ("context", "alpha"):
        roots.append(build_augmented_dataset(
            "helsinki", tmp_path / mode, levels=[2], frames=[0, 1], val_fraction=0.5,
            step_fraction=1.0, neg_stride=2, copy_paste_per_view=1, copies_per_source=1,
            sprite_review=tmp_path / "review.json", sprite_artifact_root=tmp_path,
            sprite_mode=mode, rotate_pastes=True,
        ).parent)
    labels = sorted(path.relative_to(roots[0]) for path in (roots[0] / "labels").rglob("*.txt"))
    assert labels == sorted(path.relative_to(roots[1]) for path in (roots[1] / "labels").rglob("*.txt"))
    for path in labels:
        assert (roots[0] / path).read_bytes() == (roots[1] / path).read_bytes()
    for image in (roots[0] / "images" / "val").glob("*.png"):
        assert image.read_bytes() == (roots[1] / "images" / "val" / image.name).read_bytes()
    augmented = list((roots[0] / "images" / "train").glob("*_cp*.png"))
    assert augmented
    assert any(path.read_bytes() != (roots[1] / "images" / "train" / path.name).read_bytes() for path in augmented)
