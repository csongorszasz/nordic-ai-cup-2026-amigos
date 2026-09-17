"""Utility to record incoming camera frames and metadata during validation attempts."""

import json
from pathlib import Path
from typing import Optional
import cv2

from dtos import DroneFlybyPredictRequestDto
from utils import decode_view


class ValidationDatasetRecorder:
    """Saves incoming validation frames and JSON metadata to disk for offline training."""

    def __init__(self, output_dir: Path = Path("recorded_validation_data")):
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

