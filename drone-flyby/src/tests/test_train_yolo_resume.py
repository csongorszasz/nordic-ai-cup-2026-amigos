"""Checkpoint discovery and resume behaviour of the training entrypoint."""

import argparse
from pathlib import Path

import pytest

import offline.train_yolo as train_yolo
from offline.train_yolo import build_augment_kwargs, find_last_checkpoint, train


def test_find_last_checkpoint_nested_under_runs_detect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "runs" / "detect" / "proj" / "run" / "weights"
    nested.mkdir(parents=True)
    (nested / "last.pt").write_bytes(b"x")

    assert find_last_checkpoint("proj", "run") == nested / "last.pt"


def test_find_last_checkpoint_absolute_project(tmp_path):
    weights = tmp_path / "proj" / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"x")

    assert find_last_checkpoint(str(tmp_path / "proj"), "run") == weights / "last.pt"


def test_find_last_checkpoint_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_last_checkpoint("nope", "nope") is None


def test_resume_without_checkpoint_raises(monkeypatch):
    monkeypatch.setattr(train_yolo, "_ULTRALYTICS_AVAILABLE", True)

    with pytest.raises(FileNotFoundError):
        train(
            data_yaml=Path("unused.yaml"),
            weights="unused.pt",
            epochs=1,
            imgsz=960,
            batch=1,
            device="cpu",
            resume=True,
            resume_from=None,
        )


def test_build_augment_kwargs_maps_cli_flags():
    namespace = argparse.Namespace(
        mosaic=0.0,
        mixup=0.1,
        copy_paste=0.5,
        degrees=10.0,
        scale=0.4,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        perspective=0.0,
    )

    kwargs = build_augment_kwargs(namespace)

    assert kwargs["copy_paste"] == 0.5
    assert kwargs["mosaic"] == 0.0
    assert kwargs["degrees"] == 10.0
    assert set(kwargs) == {
        "mosaic", "mixup", "copy_paste", "degrees", "scale", "translate",
        "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "erasing", "perspective",
    }


def test_train_forwards_augmentation_kwargs(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self, weights):
            self.weights = weights

        def train(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_yolo, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(train_yolo, "YOLO", FakeModel)

    train(
        data_yaml=Path("data.yaml"),
        weights="yolo11s.pt",
        epochs=1,
        imgsz=960,
        batch=2,
        device="cpu",
        augment_kwargs={"copy_paste": 0.5, "mosaic": 0.0},
    )

    assert captured["copy_paste"] == 0.5
    assert captured["mosaic"] == 0.0
    assert captured["data"] == "data.yaml"
    assert captured["imgsz"] == 960
