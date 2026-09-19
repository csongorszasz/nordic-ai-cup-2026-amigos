"""Tests for complete, request-bound, evaluation-only capture."""

import base64
import json

import pytest

from dtos import DroneFlybyPredictResponseDto, DroneFlybyPredictionDto
from offline.dataset_provenance import assert_training_source
from offline.record_dataset import ValidationDatasetRecorder, load_recorded_request
from tests.test_scaffolding import create_synthetic_request


def _request(frame_index=0, image=b"png-bytes", sequence_id="seq"):
    request = create_synthetic_request(sequence_id, frame_index)
    request.view.image = base64.b64encode(image).decode("ascii")
    return request


def _response(request):
    return DroneFlybyPredictResponseDto(
        request_id=request.request_id, frame=request.frame,
        annotations=[DroneFlybyPredictionDto(
            object_id="tank", bbox=[0.0, 0.0, 0.1, 0.1], confidence=0.9,
        )],
        requested_view=None,
    )


def test_recorder_writes_exact_replayable_inputs_responses_and_diagnostics(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path, provenance={"checkpoint_sha256": "pinned"})
    recorder.start("seq")
    request = _request()
    capture = recorder.record_frame(request)
    assert capture is not None
    recorder.record_response(
        request, _response(request), 12.5, capture=capture,
        diagnostics={"request_id": request.request_id, "raw_detections": [], "events": []},
    )
    recorder.shutdown()
    session = tmp_path / "seq"
    assert (session / "images" / f"{capture.stem}.png").read_bytes() == b"png-bytes"
    path = session / "metadata" / f"{capture.stem}.json"
    metadata = json.loads(path.read_text())
    assert metadata["source_region_xyxy"] == [0, 0, 3840, 2160]
    assert metadata["provenance"]["checkpoint_sha256"] == "pinned"
    assert metadata["request"]["original_width"] == 3840
    assert metadata["request"]["camera_constraints"] == request.camera_constraints.model_dump(mode="json")
    assert "image" not in metadata["request"]["view"]
    assert load_recorded_request(path) == request
    diagnostic = json.loads((session / "diagnostics" / f"{capture.stem}.json").read_text())
    assert diagnostic["pipeline"]["raw_detections"] == []
    assert diagnostic["evaluator_accepted"] is None
    index = [json.loads(line) for line in (session / "index.jsonl").read_text().splitlines()]
    assert index[0]["num_annotations"] == 1
    assert index[0]["elapsed_ms"] == 12.5
    status = json.loads((session / "capture_status.json").read_text())
    assert status["frames_received"] == status["frames_written"] == 1
    assert status["responses_generated"] == status["responses_written"] == 1
    assert recorder.stats()["pending_jobs"] == 0
    with pytest.raises(ValueError, match="Non-training"):
        assert_training_source(session / "images")


def test_recorder_disabled_is_a_noop(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    request = _request()
    assert recorder.record_frame(request) is None
    recorder.record_response(request, _response(request), 1.0)
    recorder.flush()
    assert not (tmp_path / "seq").exists()


def test_interleaved_sequences_keep_request_and_response_together(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    first, second = _request(sequence_id="seq_a"), _request(sequence_id="seq_b")
    recorder.start("seq_a")
    capture_a = recorder.record_frame(first)
    recorder.start("seq_b")
    capture_b = recorder.record_frame(second)
    recorder.record_response(first, _response(first), 1, capture=capture_a)
    recorder.record_response(second, _response(second), 2, capture=capture_b)
    recorder.shutdown()
    for request, capture in ((first, capture_a), (second, capture_b)):
        payload = json.loads((tmp_path / request.sequence_id / "responses" / f"{capture.stem}.json").read_text())
        assert payload["request_id"] == request.request_id


def test_duplicate_frame_indices_do_not_overwrite_inputs(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    recorder.start("seq")
    first = recorder.record_frame(_request(image=b"first"))
    second = recorder.record_frame(_request(image=b"second"))
    recorder.shutdown()
    assert first.stem != second.stem
    assert (tmp_path / "seq" / "images" / f"{first.stem}.png").read_bytes() == b"first"
    assert (tmp_path / "seq" / "images" / f"{second.stem}.png").read_bytes() == b"second"


@pytest.mark.parametrize("sequence", ["../outside", "..", "/absolute", "a/b", "a\\b", "C:\\outside", "x" * 129])
def test_recorder_rejects_unsafe_sequence_paths(tmp_path, sequence):
    recorder = ValidationDatasetRecorder(tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        recorder.start(sequence)
    assert recorder._worker is None


def test_recorder_rejects_sequence_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "recordings"
    recorder = ValidationDatasetRecorder(root)
    try:
        (root / "seq").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="escapes"):
        recorder.start("seq")
    assert not list(outside.iterdir())


def test_recorder_does_not_relabel_training_data(tmp_path):
    (tmp_path / "data_role.json").write_text(json.dumps({"data_role": "training-development"}))
    with pytest.raises(ValueError, match="another data role"):
        ValidationDatasetRecorder(tmp_path)


def test_queue_drops_and_flush_timeout_are_explicit(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path, max_queue=2)
    recorder.enabled = True
    for _ in range(5):
        recorder.record_frame(_request())
    assert recorder.dropped == 3
    with pytest.raises(TimeoutError, match="pending jobs"):
        recorder.flush(timeout=0.01)


def test_failed_write_is_persisted_and_not_reported_as_complete(tmp_path, monkeypatch):
    recorder = ValidationDatasetRecorder(tmp_path)
    def fail(*args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(recorder, "_write_frame", fail)
    recorder.start("seq")
    recorder.record_frame(_request())
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.shutdown()
    assert not recorder._worker.is_alive()
    assert recorder.stats()["write_errors"] == 1
    status = json.loads((tmp_path / "seq" / "capture_status.json").read_text())
    assert "simulated disk failure" in status["last_write_error"]


def test_response_and_diagnostics_identity_cannot_cross_requests(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    recorder.start("seq")
    request = _request()
    capture = recorder.record_frame(request)
    with pytest.raises(ValueError, match="received request"):
        recorder.record_response(_request(1), _response(_request(1)), 1, capture=capture)
    with pytest.raises(ValueError, match="another request"):
        recorder.record_response(request, _response(request), 1, capture=capture,
                                 diagnostics={"request_id": "wrong"})
    recorder.shutdown()


def test_failed_prediction_retains_input_without_faking_a_response(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    recorder.start("seq")
    request = _request()
    capture = recorder.record_frame(request)
    recorder.record_response(request, None, 1.0, capture=capture,
                             diagnostics={"request_id": request.request_id, "events": ["pipeline_error"]})
    recorder.shutdown()
    assert not (tmp_path / "seq" / "responses" / f"{capture.stem}.json").exists()
    diagnostic = json.loads((tmp_path / "seq" / "diagnostics" / f"{capture.stem}.json").read_text())
    assert diagnostic["response_generated"] is False
    assert recorder.stats()["responses_generated"] == 0


def test_replay_rejects_changed_image_bytes(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    recorder.start("seq")
    capture = recorder.record_frame(_request())
    recorder.shutdown()
    (tmp_path / "seq" / "images" / f"{capture.stem}.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_recorded_request(tmp_path / "seq" / "metadata" / f"{capture.stem}.json")
