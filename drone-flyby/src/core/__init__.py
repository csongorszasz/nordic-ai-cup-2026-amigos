"""Core processing engine for the drone-flyby task."""

from config import DEFAULT_CONFIG, DroneFlybyConfig
from core.camera_policy import create_camera_policy
from core.detector import create_detector
from core.pipeline import PipelineOrchestrator
from core.tracker import create_tracker


def build_pipeline(config: DroneFlybyConfig = DEFAULT_CONFIG) -> PipelineOrchestrator:
    """Factory helper to construct the full pipeline orchestrator from config."""
    detector = create_detector(config)
    tracker = create_tracker(config)
    camera_policy = create_camera_policy(config)
    return PipelineOrchestrator(
        detector=detector,
        tracker=tracker,
        camera_policy=camera_policy,
        config=config,
    )

