import json

import pytest

from offline.dataset_provenance import (
    assert_training_source, assert_training_yaml, mark_training_dataset, split_source_frames,
)


def test_split_groups_all_views_of_each_source_frame():
    split = split_source_frames([0, 0, 1, 2, 3, 4], 0.4)
    assert split == {0: "train", 1: "val", 2: "train", 3: "val", 4: "train"}


def test_split_does_not_reserve_all_but_one_late_class_observation():
    split = split_source_frames(list(range(25)), 0.15)
    assert sum(split[frame] == "train" for frame in range(20, 25)) >= 4


def test_evaluation_role_survives_directory_rename(tmp_path):
    source = tmp_path / "renamed"
    source.mkdir()
    (source / "data_role.json").write_text(json.dumps({"data_role": "evaluation-only"}))
    with pytest.raises(ValueError, match="Non-training"):
        assert_training_source(source / "images" / "frame.png")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("train: renamed/images\nval: val/images\n")
    with pytest.raises(ValueError, match="Non-training"):
        assert_training_yaml(yaml_path)


def test_old_unmarked_recordings_are_quarantined(tmp_path):
    source = tmp_path / "old-run"
    (source / "metadata").mkdir(parents=True)
    (source / "responses").mkdir()
    with pytest.raises(ValueError, match="Recorded evaluation"):
        assert_training_source(source / "images")


def test_dataset_build_cannot_mix_with_stale_files(tmp_path):
    directory = tmp_path / "dataset"
    mark_training_dataset(directory)
    with pytest.raises(FileExistsError):
        mark_training_dataset(directory)
