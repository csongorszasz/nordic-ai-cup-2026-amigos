"""Preflight errors for the TensorRT export helper."""

import sys
import types

import pytest

import offline.export_tensorrt as export_tensorrt


def test_export_requires_ultralytics(monkeypatch, tmp_path):
    monkeypatch.setattr(export_tensorrt, "_ULTRALYTICS_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        export_tensorrt.export(tmp_path / "w.pt", 960, True, True, 4)


def test_export_requires_tensorrt_package(monkeypatch, tmp_path):
    monkeypatch.setattr(export_tensorrt, "_ULTRALYTICS_AVAILABLE", True)
    # Force the `import tensorrt` inside export() to fail.
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    with pytest.raises(RuntimeError):
        export_tensorrt.export(tmp_path / "w.pt", 960, True, True, 4)


def test_export_requires_existing_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(export_tensorrt, "_ULTRALYTICS_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))

    with pytest.raises(FileNotFoundError):
        export_tensorrt.export(tmp_path / "missing.pt", 960, True, True, 4)
