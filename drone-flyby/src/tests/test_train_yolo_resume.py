"""Checkpoint discovery and resume behaviour of the training entrypoint."""

from pathlib import Path

import pytest

import offline.train_yolo as train_yolo
from offline.train_yolo import find_last_checkpoint, train


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
