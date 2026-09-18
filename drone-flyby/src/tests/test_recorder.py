"""Tests for the asynchronous validation recorder."""

import base64
import json
from types import SimpleNamespace

from offline.record_dataset import ValidationDatasetRecorder


def _request(frame_index: int = 0, image: bytes = b"png-bytes", feedback=None):
    view = SimpleNamespace(
        resolution_level=0,
        center_x=1920,
        center_y=1080,
        source_region_xyxy=[0, 0, 3840, 2160],
        image=base64.b64encode(image).decode("ascii"),
    )
    return SimpleNamespace(
        sequence_id="seq",
        frame=frame_index,
        frame_index=frame_index,
        request_id=f"seq:{frame_index}",
        frame_interval_ms=333,
        response_timeout_ms=3333,
        view=view,
        camera_command_feedback=feedback,
    )


class _Response:
    def __init__(self, frame: int):
        self.request_id = f"seq:{frame}"
        self.frame = frame
        self.annotations = [
            {"object_id": "tank", "bbox": [0.0, 0.0, 0.1, 0.1], "confidence": 0.9}
        ]
        self.requested_view = {"resolution_level": 1, "center_x": 1920, "center_y": 1080}

    def model_dump(self):
        return {
            "request_id": self.request_id,
            "frame": self.frame,
            "annotations": self.annotations,
            "requested_view": self.requested_view,
        }


def test_recorder_writes_inputs_and_responses(tmp_path):
    recorder = ValidationDatasetRecorder(output_dir=tmp_path)
    recorder.start("seq")

    request = _request(0)
    recorder.record_frame(request)
    recorder.record_response(request, _Response(0), 12.5)
    recorder.flush()
    recorder.shutdown()

    session = tmp_path / "seq"
    assert (session / "images" / "frame_000000.png").read_bytes() == b"png-bytes"

    metadata = json.loads((session / "metadata" / "frame_000000.json").read_text())
    assert metadata["source_region_xyxy"] == [0, 0, 3840, 2160]

    response = json.loads((session / "responses" / "frame_000000.json").read_text())
    assert response["annotations"][0]["object_id"] == "tank"
    assert response["requested_view"]["resolution_level"] == 1

    index = [json.loads(line) for line in (session / "index.jsonl").read_text().splitlines()]
    assert index[0]["num_annotations"] == 1
    assert index[0]["elapsed_ms"] == 12.5


def test_recorder_records_camera_command_feedback(tmp_path):
    recorder = ValidationDatasetRecorder(output_dir=tmp_path)
    recorder.start("seq")
    recorder.record_frame(
        _request(3, feedback={"frame": 3, "reason": "too far", "requested_view": {}})
    )
    recorder.flush()
    recorder.shutdown()

    metadata = json.loads((tmp_path / "seq" / "metadata" / "frame_000003.json").read_text())
    assert metadata["camera_command_feedback"]["reason"] == "too far"


def test_recorder_disabled_is_a_noop(tmp_path):
    recorder = ValidationDatasetRecorder(output_dir=tmp_path)
    recorder.record_frame(_request(0))
    recorder.record_response(_request(0), _Response(0), 1.0)
    recorder.flush()

    assert not (tmp_path / "seq").exists()


def test_recorder_separates_sequences(tmp_path):
    recorder = ValidationDatasetRecorder(output_dir=tmp_path)

    recorder.start("seq_a")
    recorder.record_frame(_request(0))
    recorder.flush()

    recorder.start("seq_b")
    recorder.record_frame(_request(0))
    recorder.flush()
    recorder.shutdown()

    assert (tmp_path / "seq_a" / "images" / "frame_000000.png").exists()
    assert (tmp_path / "seq_b" / "images" / "frame_000000.png").exists()


def test_recorder_counts_dropped_jobs_when_queue_full(tmp_path):
    recorder = ValidationDatasetRecorder(output_dir=tmp_path, max_queue=2)
    # Enable without starting the worker, so the bounded queue cannot drain.
    recorder.enabled = True
    recorder.session_dir = tmp_path / "seq"
    recorder.session_dir.mkdir()

    for _ in range(5):
        recorder.record_frame(_request(0))

    assert recorder.dropped == 3
