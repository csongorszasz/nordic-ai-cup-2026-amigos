import json
import hashlib

import cv2
import numpy as np
import pytest

from offline.foreground_assets import composite_sprite, load_reviewed_assets, resize_sprite
import offline.foreground_assets as assets_module


def test_premultiplied_resize_does_not_leak_transparent_source_context():
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[:, :, :3] = (0, 255, 0)
    image[0, 0] = (0, 0, 200, 255)
    sprite = resize_sprite(image, 8, 8)
    target = np.full((8, 8, 3), (100, 0, 0), dtype=np.uint8)
    composite_sprite(target, sprite)
    assert not target[:, :, 1].any()
    assert tuple(target[-1, -1]) == (100, 0, 0)
    assert target[:, :, 2].max() == 200
    assert sprite.shape[:2] == (8, 8)


def test_opaque_control_remains_pixel_identical():
    image = np.full((4, 6, 3), (20, 30, 40), dtype=np.uint8)
    target = np.zeros_like(image)
    composite_sprite(target, resize_sprite(image, 6, 4))
    np.testing.assert_array_equal(target, image)


def test_unreviewed_masks_cannot_be_loaded(tmp_path):
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"status": "unreviewed"}))
    with pytest.raises(ValueError, match="visual review"):
        load_reviewed_assets(review, tmp_path)


def test_raw_alpha_cannot_silently_bypass_correct_compositing():
    with pytest.raises(ValueError, match="premultiplied"):
        composite_sprite(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 4), np.uint8) * 255)


def test_review_pins_manifest_pixels_and_annotation_canvas(tmp_path, monkeypatch):
    monkeypatch.setattr(assets_module, "OBJECT_CLASSES", ["tank"])
    bank = tmp_path / "run" / "runs" / "foreground_bank"
    bank.mkdir(parents=True)
    image = bank / "tank.png"
    cv2.imwrite(str(image), np.full((2, 3, 4), 255, dtype=np.uint8))
    manifest = bank / "manifest.json"
    manifest.write_text(json.dumps({"assets": [{
        "object_id": "tank", "file": "tank.png",
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "source": {"scene": "training", "frame": 0, "bbox": [10, 20, 13, 22]},
    }]}))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "status": "reviewed-for-composite-audit", "accepted_classes": ["tank"],
        "default_source": "base", "annotation_convention": "source canvas",
        "sources": {"base": {
            "experiment": "run", "manifest": "runs/foreground_bank/manifest.json",
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }},
    }))
    sprites, _ = load_reviewed_assets(review, tmp_path)
    assert sprites["tank"].shape == (2, 3, 4)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pixels changed"):
        load_reviewed_assets(review, tmp_path)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        load_reviewed_assets(review, tmp_path)


def test_area_resampling_clamps_only_floating_point_alpha_roundoff(monkeypatch):
    original_resize = cv2.resize
    def rounded_resize(*args, **kwargs):
        result = original_resize(*args, **kwargs)
        result[:, :, 3] = np.nextafter(np.float32(1), np.float32(np.inf))
        return result
    monkeypatch.setattr(assets_module.cv2, "resize", rounded_resize)
    sprite = resize_sprite(np.full((13, 19, 4), 255, np.uint8), 7, 5, cv2.INTER_AREA)
    assert (sprite[:, :, 3] == 1).all()
    target = np.zeros((5, 7, 3), np.uint8)
    composite_sprite(target, sprite)
    assert target.max() == 255


def test_large_alpha_errors_are_not_hidden_by_clamping(monkeypatch):
    def invalid_resize(*args, **kwargs):
        result = np.zeros((2, 2, 4), np.float32)
        result[:, :, 3] = 1.1
        return result
    monkeypatch.setattr(assets_module.cv2, "resize", invalid_resize)
    with pytest.raises(ValueError, match="Unexpected alpha"):
        resize_sprite(np.ones((2, 2, 4), np.uint8), 2, 2, cv2.INTER_AREA)
