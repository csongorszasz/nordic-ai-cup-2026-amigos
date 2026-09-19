import pytest
import json
import numpy as np

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
