"""Central configuration and hyperparameters for the Drone-Flyby pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DroneFlybyConfig:
    # --- System & Execution ---
    DEBUG: bool = False
    DEVICE: str = "cuda:0"  # or "cpu"
    
    # --- Detector Settings ---
    DETECTOR_TYPE: str = "dummy"  # "dummy", "yolo_standard", "yolo_sahi", "tensorrt"
    YOLO_WEIGHTS_PATH: Optional[Path] = None
    TRT_ENGINE_PATH: Optional[Path] = None
    CONFIDENCE_THRESHOLD_L0: float = 0.10
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
    POLICY_TYPE: str = "sweep"  # "sweep", "survey_zoom", "belief_map"
    SURVEY_INTERVAL_FRAMES: int = 10
    
    # --- Geometry Constants ---
    SOURCE_WIDTH: int = 3840
    SOURCE_HEIGHT: int = 2160
    VIEW_WIDTH: int = 960
    VIEW_HEIGHT: int = 540


# Global default configuration instance
DEFAULT_CONFIG = DroneFlybyConfig()

