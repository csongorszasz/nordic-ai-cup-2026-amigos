import importlib.util
import json
from pathlib import Path

import pytest

from config import DroneFlybyConfig
import core
from offline import record_dataset
from tests.test_scaffolding import create_synthetic_request


def _api(tmp_path, monkeypatch):
    pipeline = core.build_pipeline(DroneFlybyConfig(
        DETECTOR_TYPE="dummy", POLICY_TYPE="hold", TRACKER_TYPE="world_map",
    ))
    recorder = record_dataset.ValidationDatasetRecorder(tmp_path)
    monkeypatch.setattr(core, "build_pipeline", lambda config: pipeline)
    monkeypatch.setattr(record_dataset, "recorder_from_env", lambda: recorder)
    spec = importlib.util.spec_from_file_location("api_capture_test", Path(__file__).resolve().parents[1] / "api.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api, recorder


def test_api_captures_the_exact_request_with_matching_pipeline_diagnostics(tmp_path, monkeypatch):
    api, recorder = _api(tmp_path, monkeypatch)
    request = create_synthetic_request("api_capture")
    response = api.predict_endpoint(request)
    recorder.shutdown()
    metadata = next((tmp_path / "api_capture" / "metadata").glob("*.json"))
    assert record_dataset.load_recorded_request(metadata) == request
    diagnostic = json.loads((tmp_path / "api_capture" / "diagnostics" / metadata.name).read_text())
    assert diagnostic["pipeline"]["request_id"] == response.request_id
    assert diagnostic["pipeline"]["raw_detections"] is not None
    assert len(diagnostic["pipeline"]["output_sources"]) == len(response.annotations)
    assert recorder.stats()["write_errors"] == 0


def test_recording_failure_does_not_break_valid_prediction(tmp_path, monkeypatch):
    api, recorder = _api(tmp_path, monkeypatch)
    def fail(*args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(recorder, "_write_frame", fail)
    request = create_synthetic_request("disk_failure")
    response = api.predict_endpoint(request)
    assert response.request_id == request.request_id
    assert response.annotations
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.shutdown()
    assert recorder.stats()["write_errors"] > 0


def test_unsafe_capture_path_is_rejected_without_rejecting_protocol_request(tmp_path, monkeypatch):
    api, recorder = _api(tmp_path, monkeypatch)
    request = create_synthetic_request("../unsafe")
    response = api.predict_endpoint(request)
    assert response.request_id == request.request_id
    assert recorder.stats()["capture_errors"] == 1
    with pytest.raises(RuntimeError, match="incomplete"):
        recorder.shutdown()
    assert not (tmp_path.parent / "unsafe").exists()


def test_response_validation_error_remains_an_error_and_preserves_input(tmp_path, monkeypatch):
    api, recorder = _api(tmp_path, monkeypatch)
    def fail(response):
        raise ValueError("simulated invalid response")
    monkeypatch.setattr(api, "validate_response", fail)
    with pytest.raises(ValueError, match="simulated invalid"):
        api.predict_endpoint(create_synthetic_request("invalid_response"))
    recorder.shutdown()
    path = next((tmp_path / "invalid_response" / "diagnostics").glob("*.json"))
    diagnostic = json.loads(path.read_text())
    assert diagnostic["response_generated"] is False
    assert "response_validation_error" in diagnostic["pipeline"]["events"]
    assert list((tmp_path / "invalid_response" / "images").glob("*.png"))
    assert not list((tmp_path / "invalid_response" / "responses").glob("*.json"))
