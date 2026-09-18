"""Configuration defaults and environment overrides."""

from pathlib import Path

from config import DEFAULT_WEIGHTS_PATH, DroneFlybyConfig


def test_default_weights_path_is_absolute_and_under_weights():
    path = Path(DEFAULT_WEIGHTS_PATH)
    assert path.is_absolute()
    assert path.parent.name == "weights"
    assert path.name == "yolo11s_drone_flyby.pt"


def test_env_overrides_detector_weights_and_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv("DRONE_FLYBY_DETECTOR_TYPE", "dummy")
    monkeypatch.setenv("DRONE_FLYBY_YOLO_WEIGHTS_PATH", str(tmp_path / "w.pt"))
    monkeypatch.setenv("DRONE_FLYBY_CALIBRATION_PATH", str(tmp_path / "cal.json"))

    config = DroneFlybyConfig.from_env()

    assert config.DETECTOR_TYPE == "dummy"
    assert config.YOLO_WEIGHTS_PATH == tmp_path / "w.pt"
    assert config.CALIBRATION_PATH == tmp_path / "cal.json"


def test_env_unset_keeps_defaults(monkeypatch):
    for name in (
        "DRONE_FLYBY_DETECTOR_TYPE",
        "DRONE_FLYBY_YOLO_WEIGHTS_PATH",
        "DRONE_FLYBY_CALIBRATION_PATH",
        "DRONE_FLYBY_TRACKER_TYPE",
        "DRONE_FLYBY_POLICY_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = DroneFlybyConfig.from_env()

    assert config.DETECTOR_TYPE == "yolo_standard"
    assert config.YOLO_WEIGHTS_PATH == DroneFlybyConfig.YOLO_WEIGHTS_PATH
    assert config.CALIBRATION_PATH is None


def test_env_blank_path_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DRONE_FLYBY_YOLO_WEIGHTS_PATH", "   ")

    config = DroneFlybyConfig.from_env()

    assert config.YOLO_WEIGHTS_PATH == DroneFlybyConfig.YOLO_WEIGHTS_PATH
