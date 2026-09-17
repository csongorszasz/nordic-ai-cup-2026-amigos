"""Tests for dataset building and deployment configuration helpers."""

from pathlib import Path

from config import DroneFlybyConfig
from offline.train_yolo import build_dataset_layout
from utils import DATA_DIRECTORY


def test_config_from_env_overrides_detector_and_weights(monkeypatch):
    monkeypatch.setenv("DRONE_FLYBY_DETECTOR_TYPE", "yolo_standard")
    monkeypatch.setenv("DRONE_FLYBY_YOLO_WEIGHTS_PATH", "/tmp/best.pt")
    monkeypatch.setenv("DRONE_FLYBY_POLICY_TYPE", "survey_zoom")
    monkeypatch.setenv("DRONE_FLYBY_TRACKER_TYPE", "world_map")

    config = DroneFlybyConfig.from_env()

    assert config.DETECTOR_TYPE == "yolo_standard"
    assert config.YOLO_WEIGHTS_PATH == Path("/tmp/best.pt")
    assert config.POLICY_TYPE == "survey_zoom"
    assert config.TRACKER_TYPE == "world_map"


def test_build_dataset_layout_creates_train_and_val_splits(tmp_path):
    helsinki_dir = DATA_DIRECTORY / "helsinki"
    output_dir = tmp_path / "artifacts"

    data_yaml = build_dataset_layout(output_dir, helsinki_dir, validation_split=0.2)

    dataset_dir = output_dir / "drone_flyby_dataset"
    assert data_yaml.exists()
    assert (dataset_dir / "images" / "train").exists()
    assert (dataset_dir / "images" / "val").exists()
    assert (dataset_dir / "labels" / "train").exists()
    assert (dataset_dir / "labels" / "val").exists()

    train_images = list((dataset_dir / "images" / "train").glob("*.png"))
    val_images = list((dataset_dir / "images" / "val").glob("*.png"))
    train_labels = list((dataset_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((dataset_dir / "labels" / "val").glob("*.txt"))

    assert train_images
    assert val_images
    assert train_labels
    assert val_labels
    assert len(train_images) + len(val_images) == 25
    assert len(train_labels) == len(train_images)
    assert len(val_labels) == len(val_images)
