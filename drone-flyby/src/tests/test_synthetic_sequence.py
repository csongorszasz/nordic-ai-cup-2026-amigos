import pytest
import json
import hashlib
import numpy as np
from PIL import Image

from offline.build_synthetic_sequence import build_sequence, project_annotations
import offline.build_synthetic_sequence as generator
from dtos import OBJECT_CLASSES


def test_projection_preserves_identity_and_source_motion():
    objects = [{"object_id": "tank", "instance_id": "one", "bbox": [20, 30, 60, 70]}]
    first = project_annotations(objects, (0, 20), width=100, height=100)
    next_frame = project_annotations(objects, (-10, 0), width=100, height=100)
    assert first[0]["bbox"] == [20, 10, 60, 50]
    assert next_frame[0]["bbox"] == [30, 30, 70, 70]
    assert first[0]["instance_id"] == next_frame[0]["instance_id"]


def test_projection_clips_visible_edges_and_drops_invisible_objects():
    objects = [{"object_id": "tank", "instance_id": "one", "bbox": [20, 30, 60, 70]}]
    assert project_annotations(objects, (0, 50), width=100, height=100)[0]["bbox"] == [20, 0, 60, 20]
    assert project_annotations(objects, (0, 70), width=100, height=100) == []


def test_existing_episode_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        build_sequence(tmp_path)


def test_rotation_ablation_uses_identical_placement_and_scale(tmp_path, monkeypatch):
    sprites = {name: np.full((12, 24, 3), 220, dtype=np.uint8) for name in OBJECT_CLASSES}
    monkeypatch.setattr(generator, "harvest_sprites", lambda scene: (sprites, {}))
    for rotate in (False, True):
        build_sequence(tmp_path / str(rotate), frames=2, objects_per_frame=2, rotate=rotate)
    native = json.loads((tmp_path / "False" / "run_metadata.json").read_text())
    rotated = json.loads((tmp_path / "True" / "run_metadata.json").read_text())
    for first, second in zip(native["objects"], rotated["objects"]):
        assert first["placement"] == second["placement"]
        assert first["scale"] == second["scale"]
        assert first["rotation_quarters"] == 0


def test_continuous_rotation_audit_preserves_centers_layout_scale_and_background(tmp_path, monkeypatch):
    image = np.full((12, 25, 4), (0, 255, 0, 0), np.uint8)
    image[2:10, 4:21] = (0, 0, 200, 255)
    sprites = {name: image.copy() for name in OBJECT_CLASSES}
    monkeypatch.setattr(generator, "load_reviewed_assets", lambda *args: (sprites, {}))
    controls = []
    for offset in (0, 22.5, 45):
        output = tmp_path / str(offset)
        build_sequence(
            output, frames=2, objects_per_frame=2, seed=919,
            sprite_review=tmp_path / "review.json", sprite_artifact_root=tmp_path,
            rotation_offset_degrees=offset,
        )
        controls.append(json.loads((output / "run_metadata.json").read_text()))
    for variant in controls[1:]:
        assert variant["background"] == controls[0]["background"]
        for reference, current in zip(controls[0]["objects"], variant["objects"]):
            assert reference["placement"] == current["placement"]
            assert reference["scale"] == current["scale"]
            assert reference["rotation_quarters"] == current["rotation_quarters"]
            assert reference["object_id"] == current["object_id"]
            a, b = reference["bbox"], current["bbox"]
            assert (a[0] + a[2], a[1] + a[3]) == pytest.approx((b[0] + b[2], b[1] + b[3]))


def test_continuous_rotation_requires_reviewed_alpha_assets(tmp_path):
    with pytest.raises(ValueError, match="reviewed alpha"):
        build_sequence(tmp_path / "unreviewed", frames=2, rotation_offset_degrees=30)


def test_orthophoto_uses_verified_scale_and_preserves_attribution(tmp_path, monkeypatch):
    image = tmp_path / "background.png"
    Image.fromarray(np.full((64, 64, 3), [20, 80, 120], dtype=np.uint8)).save(image)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({
        "source_url": "https://example.invalid/test-image",
        "license": "fixture", "attribution": "fixture provider",
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr(generator, "read_georeference", lambda path: {"gsd_m": 0.5, "gsd_source": "embedded-gml"})
    terrain, metadata = generator.orthophoto_background(
        image, manifest, 64, 32, np.random.default_rng(0), target_gsd=0.25, origin=(8, 9),
    )
    assert terrain.shape == (32, 64, 3)
    assert tuple(terrain[0, 0]) == (120, 80, 20)
    assert metadata["output_gsd_x_m"] == 0.25
    assert metadata["output_gsd_y_m"] == 0.25
    assert metadata["attribution"] == "fixture provider"
    assert metadata["source_crop_xyxy"] == [8, 9, 40, 25]
    with pytest.raises(ValueError, match="outside"):
        generator.orthophoto_background(image, manifest, 64, 32, np.random.default_rng(0),
                                       target_gsd=0.25, origin=(63, 63))


def test_orthophoto_requires_provenance_and_matching_hash(tmp_path):
    image = tmp_path / "background.jp2"
    image.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="provenance"):
        generator.orthophoto_background(image, None, 64, 32, np.random.default_rng(0))
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({
        "source_url": "fixture", "license": "fixture", "attribution": "fixture", "sha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="hash"):
        generator.orthophoto_background(image, manifest, 64, 32, np.random.default_rng(0))
