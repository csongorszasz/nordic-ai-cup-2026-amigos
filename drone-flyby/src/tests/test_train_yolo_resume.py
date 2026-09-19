"""Checkpoint discovery and resume behaviour of the training entrypoint."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_train_forwards_augmentation_kwargs(monkeypatch, tmp_path):
    captured = {}

    class FakeModel:
        task = "detect"

        def __init__(self, weights):
            self.weights = weights

        def train(self, **kwargs):
            captured.update(kwargs)

        def add_callback(self, event, callback):
            assert event == "on_fit_epoch_end"

    monkeypatch.setattr(train_yolo, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(train_yolo, "YOLO", FakeModel)

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\n")
    train(
        data_yaml=data_yaml,
        weights="yolo11s.pt",
        epochs=1,
        imgsz=960,
        batch=2,
        device="cpu",
        augment_kwargs={"mixup": 0.1, "mosaic": 0.0},
    )

    assert captured["mixup"] == 0.1
    assert captured["mosaic"] == 0.0
    assert captured["data"] == str(data_yaml)
    assert captured["imgsz"] == 960
    assert captured["seed"] == 0
    assert captured["save_period"] == 10
    assert captured["warmup_epochs"] == 3.0


def test_ap50_checkpoint_selection_is_independent_of_library_fitness(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    last = weights / "last.pt"
    trainer = SimpleNamespace(save_dir=tmp_path, last=last, epoch=0,
                              metrics={"metrics/mAP50(B)": 0.7})
    last.write_bytes(b"first")
    train_yolo.save_ap50_checkpoint(trainer)
    last.write_bytes(b"worse-ap50")
    trainer.metrics["metrics/mAP50(B)"] = 0.6
    train_yolo.save_ap50_checkpoint(trainer)
    assert (weights / "best_ap50.pt").read_bytes() == b"first"
    trainer.metrics["metrics/mAP50(B)"] = 0.8
    trainer.epoch = 2
    last.write_bytes(b"better-ap50")
    train_yolo.save_ap50_checkpoint(trainer)
    assert (weights / "best_ap50.pt").read_bytes() == b"better-ap50"
    assert json.loads((weights / "best_ap50.json").read_text())["epoch"] == 3


def test_training_rejects_silent_segmentation_only_copy_paste(monkeypatch, tmp_path):
    class FakeModel:
        task = "detect"

        def __init__(self, weights):
            pass

    monkeypatch.setattr(train_yolo, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setattr(train_yolo, "YOLO", FakeModel)
    yaml = tmp_path / "data.yaml"
    yaml.write_text("train: images/train\nval: images/val\n")
    with pytest.raises(ValueError, match="segmentation masks"):
        train(yaml, "fake.pt", 1, 960, 1, "cpu", augment_kwargs={"copy_paste": 0.5})
