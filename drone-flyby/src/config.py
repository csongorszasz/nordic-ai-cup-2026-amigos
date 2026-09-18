"""Central configuration and hyperparameters for the Drone-Flyby pipeline."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


# The task-finetuned weights shipped with the repository. The default detector
# is a real 16-class detector rather than the template baseline: a missing
# weight file must fail startup, not silently change the model.
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "weights" / "yolo11s_drone_flyby.pt"
DEFAULT_TRT_ENGINE_PATH = Path(__file__).resolve().parents[1] / "weights" / "yolo11s_drone_flyby.engine"


@dataclass
class DroneFlybyConfig:
    # --- System & Execution ---
    DEBUG: bool = False
    DEVICE: str = "cuda:0"  # or "cpu"
    
    # --- Detector Settings ---
    DETECTOR_TYPE: str = "yolo_standard"  # "yolo_standard", "tensorrt"; "dummy"/"template_bank" are debug-only
    YOLO_WEIGHTS_PATH: Optional[Path] = DEFAULT_WEIGHTS_PATH
    TRT_ENGINE_PATH: Optional[Path] = DEFAULT_TRT_ENGINE_PATH
    # Optional per-class confidence calibration produced by offline/calibrate.py.
    CALIBRATION_PATH: Optional[Path] = None
    CONFIDENCE_THRESHOLD_L0: float = 0.25
    CONFIDENCE_THRESHOLD_L1: float = 0.15
    CONFIDENCE_THRESHOLD_L2: float = 0.20
    
    # --- Tracker & Spatial Memory Settings ---
    # Defaults select the recommended architecture: a persistent world map fed
    # by coverage-first active vision. The previous passthrough/hold pair was a
    # score-honest baseline, but it holds the camera at Level 0 where the
    # objects are a few pixels, so it scores zero by construction.
    TRACKER_TYPE: str = "world_map"  # "passthrough", "world_map"
    IOU_MATCH_THRESHOLD: float = 0.30
    MIN_HITS_TO_CONFIRM: int = 2
    SINGLE_HIT_CONFIRM_CONFIDENCE: float = 0.30
    CONFIDENCE_DECAY_RATE: float = 0.98  # Slow decay for static objects
    OUT_OF_VIEW_MAX_AGE_FRAMES: int = 50
    MIN_EXISTENCE: float = 0.20
    GLOBAL_NMS_IOU_THRESHOLD: float = 0.45

    # --- Ego-motion ---
    # "phase_correlation" is cheap and proven; "ecc" is more robust on
    # low-texture or brightness-varying views.
    EGO_MOTION_METHOD: str = "phase_correlation"
    EGO_MOTION_MIN_RESPONSE: float = 0.15

    # --- Camera Policy Settings ---
    POLICY_TYPE: str = "belief_voi"  # "hold", "sweep", "survey_zoom", "deterministic_l1", "active_coverage", "belief_voi"
    SURVEY_INTERVAL_FRAMES: int = 4

    # --- Belief-map Value-of-Information Planner ---
    BELIEF_CELL_SIZE: int = 120
    BELIEF_EXPLORE_WEIGHT: float = 1.0
    BELIEF_VERIFY_WEIGHT: float = 1.2
    BELIEF_TRAVEL_WEIGHT: float = 0.35
    BELIEF_L2_MIN_INTERVAL: int = 4
    
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

        def _get_path(name: str, default: Optional[Path]) -> Optional[Path]:
            value = os.getenv(name)
            return default if value is None or not value.strip() else Path(value.strip())

        return cls(
            DEBUG=_get_bool("DRONE_FLYBY_DEBUG", cls.DEBUG),
            DEVICE=_get_str("DRONE_FLYBY_DEVICE", cls.DEVICE),
            DETECTOR_TYPE=_get_str("DRONE_FLYBY_DETECTOR_TYPE", cls.DETECTOR_TYPE),
            YOLO_WEIGHTS_PATH=_get_path("DRONE_FLYBY_YOLO_WEIGHTS_PATH", cls.YOLO_WEIGHTS_PATH),
            TRT_ENGINE_PATH=_get_path("DRONE_FLYBY_TRT_ENGINE_PATH", cls.TRT_ENGINE_PATH),
            CALIBRATION_PATH=_get_path("DRONE_FLYBY_CALIBRATION_PATH", cls.CALIBRATION_PATH),
            CONFIDENCE_THRESHOLD_L0=_get_float("DRONE_FLYBY_CONF_L0", cls.CONFIDENCE_THRESHOLD_L0),
            CONFIDENCE_THRESHOLD_L1=_get_float("DRONE_FLYBY_CONF_L1", cls.CONFIDENCE_THRESHOLD_L1),
            CONFIDENCE_THRESHOLD_L2=_get_float("DRONE_FLYBY_CONF_L2", cls.CONFIDENCE_THRESHOLD_L2),
            TRACKER_TYPE=_get_str("DRONE_FLYBY_TRACKER_TYPE", cls.TRACKER_TYPE),
            IOU_MATCH_THRESHOLD=_get_float("DRONE_FLYBY_IOU_MATCH_THRESHOLD", cls.IOU_MATCH_THRESHOLD),
            MIN_HITS_TO_CONFIRM=_get_int("DRONE_FLYBY_MIN_HITS_TO_CONFIRM", cls.MIN_HITS_TO_CONFIRM),
            SINGLE_HIT_CONFIRM_CONFIDENCE=_get_float(
                "DRONE_FLYBY_SINGLE_HIT_CONFIRM_CONFIDENCE", cls.SINGLE_HIT_CONFIRM_CONFIDENCE
            ),
            CONFIDENCE_DECAY_RATE=_get_float("DRONE_FLYBY_CONFIDENCE_DECAY_RATE", cls.CONFIDENCE_DECAY_RATE),
            OUT_OF_VIEW_MAX_AGE_FRAMES=_get_int("DRONE_FLYBY_OUT_OF_VIEW_MAX_AGE_FRAMES", cls.OUT_OF_VIEW_MAX_AGE_FRAMES),
            MIN_EXISTENCE=_get_float("DRONE_FLYBY_MIN_EXISTENCE", cls.MIN_EXISTENCE),
            GLOBAL_NMS_IOU_THRESHOLD=_get_float("DRONE_FLYBY_GLOBAL_NMS_IOU_THRESHOLD", cls.GLOBAL_NMS_IOU_THRESHOLD),
            EGO_MOTION_METHOD=_get_str("DRONE_FLYBY_EGO_MOTION_METHOD", cls.EGO_MOTION_METHOD),
            EGO_MOTION_MIN_RESPONSE=_get_float("DRONE_FLYBY_EGO_MOTION_MIN_RESPONSE", cls.EGO_MOTION_MIN_RESPONSE),
            POLICY_TYPE=_get_str("DRONE_FLYBY_POLICY_TYPE", cls.POLICY_TYPE),
            SURVEY_INTERVAL_FRAMES=_get_int("DRONE_FLYBY_SURVEY_INTERVAL_FRAMES", cls.SURVEY_INTERVAL_FRAMES),
            BELIEF_CELL_SIZE=_get_int("DRONE_FLYBY_BELIEF_CELL_SIZE", cls.BELIEF_CELL_SIZE),
            BELIEF_EXPLORE_WEIGHT=_get_float("DRONE_FLYBY_BELIEF_EXPLORE_WEIGHT", cls.BELIEF_EXPLORE_WEIGHT),
            BELIEF_VERIFY_WEIGHT=_get_float("DRONE_FLYBY_BELIEF_VERIFY_WEIGHT", cls.BELIEF_VERIFY_WEIGHT),
            BELIEF_TRAVEL_WEIGHT=_get_float("DRONE_FLYBY_BELIEF_TRAVEL_WEIGHT", cls.BELIEF_TRAVEL_WEIGHT),
            BELIEF_L2_MIN_INTERVAL=_get_int("DRONE_FLYBY_BELIEF_L2_MIN_INTERVAL", cls.BELIEF_L2_MIN_INTERVAL),
        )


# Global default configuration instance
DEFAULT_CONFIG = DroneFlybyConfig.from_env()

