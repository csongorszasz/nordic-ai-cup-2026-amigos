"""Central configuration and hyperparameters for the Drone-Flyby pipeline."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


@dataclass
class DroneFlybyConfig:
    # --- System & Execution ---
    DEBUG: bool = False
    DEVICE: str = "cuda:0"  # or "cpu"
    
    # --- Detector Settings ---
    DETECTOR_TYPE: str = "template_bank"  # "dummy", "template_bank", "yolo_standard", "yolo_sahi", "tensorrt"
    YOLO_WEIGHTS_PATH: Optional[Path] = None
    TRT_ENGINE_PATH: Optional[Path] = None
    CONFIDENCE_THRESHOLD_L0: float = 0.25
    CONFIDENCE_THRESHOLD_L1: float = 0.15
    CONFIDENCE_THRESHOLD_L2: float = 0.20
    
    # --- Tracker & Spatial Memory Settings ---
    TRACKER_TYPE: str = "world_map"  # "passthrough", "world_map"
    IOU_MATCH_THRESHOLD: float = 0.30
    MIN_HITS_TO_CONFIRM: int = 2
    CONFIDENCE_DECAY_RATE: float = 0.98  # Slow decay for static objects
    OUT_OF_VIEW_MAX_AGE_FRAMES: int = 50
    GLOBAL_NMS_IOU_THRESHOLD: float = 0.45
    
    # --- Camera Policy Settings ---
    POLICY_TYPE: str = "survey_zoom"  # "sweep", "survey_zoom", "belief_map"
    SURVEY_INTERVAL_FRAMES: int = 4
    
    # --- Geometry Constants ---
    SOURCE_WIDTH: int = 3840
    SOURCE_HEIGHT: int = 2160
    VIEW_WIDTH: int = 960
    VIEW_HEIGHT: int = 540

    @classmethod
    def from_env(cls) -> "DroneFlybyConfig":
        """Build a config from environment variables for easy deployment."""

        def _get_bool(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        def _get_int(name: str, default: int) -> int:
            value = os.getenv(name)
            return default if value is None else int(value)

        def _get_float(name: str, default: float) -> float:
            value = os.getenv(name)
            return default if value is None else float(value)

        def _get_str(name: str, default: str) -> str:
            value = os.getenv(name)
            return default if value is None or not value.strip() else value.strip()

        def _get_path(name: str) -> Optional[Path]:
            value = os.getenv(name)
            return None if value is None or not value.strip() else Path(value.strip())

        return cls(
            DEBUG=_get_bool("DRONE_FLYBY_DEBUG", cls.DEBUG),
            DEVICE=_get_str("DRONE_FLYBY_DEVICE", cls.DEVICE),
            DETECTOR_TYPE=_get_str("DRONE_FLYBY_DETECTOR_TYPE", cls.DETECTOR_TYPE),
            YOLO_WEIGHTS_PATH=_get_path("DRONE_FLYBY_YOLO_WEIGHTS_PATH"),
            TRT_ENGINE_PATH=_get_path("DRONE_FLYBY_TRT_ENGINE_PATH"),
            CONFIDENCE_THRESHOLD_L0=_get_float("DRONE_FLYBY_CONF_L0", cls.CONFIDENCE_THRESHOLD_L0),
            CONFIDENCE_THRESHOLD_L1=_get_float("DRONE_FLYBY_CONF_L1", cls.CONFIDENCE_THRESHOLD_L1),
            CONFIDENCE_THRESHOLD_L2=_get_float("DRONE_FLYBY_CONF_L2", cls.CONFIDENCE_THRESHOLD_L2),
            TRACKER_TYPE=_get_str("DRONE_FLYBY_TRACKER_TYPE", cls.TRACKER_TYPE),
            IOU_MATCH_THRESHOLD=_get_float("DRONE_FLYBY_IOU_MATCH_THRESHOLD", cls.IOU_MATCH_THRESHOLD),
            MIN_HITS_TO_CONFIRM=_get_int("DRONE_FLYBY_MIN_HITS_TO_CONFIRM", cls.MIN_HITS_TO_CONFIRM),
            CONFIDENCE_DECAY_RATE=_get_float("DRONE_FLYBY_CONFIDENCE_DECAY_RATE", cls.CONFIDENCE_DECAY_RATE),
            OUT_OF_VIEW_MAX_AGE_FRAMES=_get_int("DRONE_FLYBY_OUT_OF_VIEW_MAX_AGE_FRAMES", cls.OUT_OF_VIEW_MAX_AGE_FRAMES),
            GLOBAL_NMS_IOU_THRESHOLD=_get_float("DRONE_FLYBY_GLOBAL_NMS_IOU_THRESHOLD", cls.GLOBAL_NMS_IOU_THRESHOLD),
            POLICY_TYPE=_get_str("DRONE_FLYBY_POLICY_TYPE", cls.POLICY_TYPE),
            SURVEY_INTERVAL_FRAMES=_get_int("DRONE_FLYBY_SURVEY_INTERVAL_FRAMES", cls.SURVEY_INTERVAL_FRAMES),
        )


# Global default configuration instance
DEFAULT_CONFIG = DroneFlybyConfig.from_env()

