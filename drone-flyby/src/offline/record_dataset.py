"""Utility to record incoming camera frames and metadata during validation attempts."""

import json
import os
from pathlib import Path
from typing import Optional
import cv2

from dtos import DroneFlybyPredictRequestDto
from utils import PROJECT_ROOT, decode_view


class ValidationDatasetRecorder:
    """Saves incoming validation frames and JSON metadata to disk for offline training."""

    def __init__(self, output_dir: Path = PROJECT_ROOT / "recorded_validation_data"):
        self.output_dir = output_dir
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.enabled = False

    def start(self, sequence_id: str) -> None:
        """Initialize directory structure for a recording session."""
        self.session_dir = self.output_dir / sequence_id
        self.images_dir = self.session_dir / "images"
        self.metadata_dir = self.session_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True

    def stop(self) -> None:
        """Disable recording for the current session."""
        self.enabled = False

    def record_frame(self, request: DroneFlybyPredictRequestDto) -> None:
        """Save image and request metadata for the current frame."""
        if not self.enabled:
            return

        frame_idx = request.frame_index
        image_path = self.images_dir / f"frame_{frame_idx:06d}.png"
        meta_path = self.metadata_dir / f"frame_{frame_idx:06d}.json"

        # Save decoded image
        image_bgr = decode_view(request.view)
        cv2.imwrite(str(image_path), image_bgr)

        # Save metadata
        meta = {
            "sequence_id": request.sequence_id,
            "frame": request.frame,
            "frame_index": request.frame_index,
            "request_id": request.request_id,
            "resolution_level": request.view.resolution_level,
            "center_x": request.view.center_x,
            "center_y": request.view.center_y,
            "source_region_xyxy": request.view.source_region_xyxy,
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)


def recorder_from_env() -> Optional[ValidationDatasetRecorder]:
    """Build a recorder when validation capture is enabled via environment."""
    enabled_value = os.getenv("DRONE_FLYBY_RECORD_VALIDATION_DATA", "0").strip().lower()
    if enabled_value not in {"1", "true", "yes", "on"}:
        return None

    output_dir = Path(os.getenv("DRONE_FLYBY_RECORD_DIR", str(PROJECT_ROOT / "recorded_validation_data")))
    return ValidationDatasetRecorder(output_dir=output_dir)

