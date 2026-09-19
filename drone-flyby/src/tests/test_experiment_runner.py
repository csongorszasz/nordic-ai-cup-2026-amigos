import argparse
import json
from pathlib import Path

import pytest
from config import DroneFlybyConfig
from core import build_pipeline
from offline.record_dataset import ValidationDatasetRecorder
from tests.test_scaffolding import create_synthetic_request

from offline import experiment_runner as runner


def test_matrix_preserves_failures_instead_of_reporting_a_score(tmp_path, monkeypatch):
    arguments = argparse.Namespace(
        mode="oracle", weights=None, output=tmp_path / "run", scene="synthetic",
        policies=["hold"], trackers=["world_map"], min_view_pixels=0,
    )
    def fail(*args, **kwargs):
        raise ValueError("broken scorer")
    monkeypatch.setattr(runner, "run_simulation", fail)
    with pytest.raises(ValueError, match="broken scorer"):
        runner.run_matrix(arguments)
    failure = json.loads((arguments.output / "world_map_hold" / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert not (arguments.output / "summary.json").exists()


def test_real_matrix_requires_an_explicit_checkpoint(tmp_path):
    arguments = argparse.Namespace(mode="http", weights=None, output=tmp_path / "run")
    with pytest.raises(ValueError, match="weights"):
        runner.run_matrix(arguments)
    assert not arguments.output.exists()


def test_rescore_converts_json_keys_and_keeps_the_full_denominator(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "scene": "fixture", "evaluation_frames": [0, 1, 2],
        "predictions": {"0": []}, "map50": 0.25,
    }))
    def score(scene, predictions, evaluation_frames):
        assert scene == "fixture"
        assert set(predictions) == {0}
        assert evaluation_frames == [0, 1, 2]
        return 0.25, {}
    monkeypatch.setattr(runner, "score", score)
    assert runner.rescore_result(path) == 0.25


def test_rescore_rejects_changed_scores(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"scene": "fixture", "predictions": {}, "map50": 0.5}))
    monkeypatch.setattr(runner, "score", lambda *args, **kwargs: (0.0, {}))
    with pytest.raises(ValueError, match="do not reproduce"):
        runner.rescore_result(path)


@pytest.mark.parametrize("mode, delay", [("oracle", 10), ("http", -1), ("http", float("nan"))])
def test_latency_stress_is_explicit_and_validated(tmp_path, mode, delay):
    arguments = argparse.Namespace(
        mode=mode, weights=tmp_path / "model.pt", output=tmp_path / "run",
        simulate_latency_ms=delay,
    )
    with pytest.raises(ValueError, match="Simulated latency"):
        runner.run_matrix(arguments)
    assert not arguments.output.exists()


def test_capture_cannot_silently_apply_to_non_http_experiments(tmp_path):
    arguments = argparse.Namespace(
        mode="oracle", weights=None, output=tmp_path / "run", capture_inputs=True,
    )
    with pytest.raises(ValueError, match="capture requires HTTP"):
        runner.run_matrix(arguments)
    assert not arguments.output.exists()


def _recorded_sequence(tmp_path):
    config = DroneFlybyConfig(DETECTOR_TYPE="dummy", POLICY_TYPE="hold", TRACKER_TYPE="world_map")
    pipeline = build_pipeline(config)
    pipeline.warmup()
    recorder = ValidationDatasetRecorder(tmp_path, provenance={"run_nonce": "fixture"})
    recorder.start("recorded")
    for index in (0, 2, 3):
        request = create_synthetic_request("recorded", index)
        capture = recorder.record_frame(request)
        diagnostics = {}
        response = pipeline.handle_request(request, diagnostics=diagnostics)
        recorder.record_response(request, response, 1.0, capture=capture, diagnostics=diagnostics)
    recorder.shutdown()
    return config, tmp_path / "recorded"


def test_recorded_views_reproduce_state_without_inventing_ap(tmp_path):
    config, recording = _recorded_sequence(tmp_path)
    result = runner.run_recorded_views(config, recording, expected_frames=4)
    assert result["request_count"] == 3
    assert result["coverage"]["missing_count"] == 1
    assert result["all_recorded_responses_match"] is True
    assert result["score_available"] is False
    assert "map50" not in result
    assert [entry["frame_index"] for entry in result["replayed_requests"]] == [0, 2, 3]


def test_recorded_views_validate_images_before_loading_a_model(tmp_path, monkeypatch):
    config, recording = _recorded_sequence(tmp_path)
    image = next((recording / "images").glob("*.png"))
    image.write_bytes(b"corrupted")
    def unexpected_model(*args):
        raise AssertionError("Do not load a model for corrupt capture")
    monkeypatch.setattr(runner, "build_pipeline", unexpected_model)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run_recorded_views(config, recording)


def test_recorded_views_reject_unverified_processing_order(tmp_path):
    config, recording = _recorded_sequence(tmp_path)
    path = next((recording / "diagnostics").glob("*.json"))
    diagnostic = json.loads(path.read_text())
    diagnostic["pipeline"].pop("processing_order")
    path.write_text(json.dumps(diagnostic))
    with pytest.raises(ValueError, match="processing order"):
        runner.run_recorded_views(config, recording)


def test_recorded_views_cannot_claim_counterfactual_camera_score(tmp_path):
    arguments = argparse.Namespace(
        mode="recording", weights=Path("model.pt"), recording=tmp_path, scene=None,
        policies=["belief_voi"], output=tmp_path / "result",
    )
    with pytest.raises(ValueError, match="policies hold"):
        runner.run_matrix(arguments)
    assert not arguments.output.exists()


def test_fixed_view_replay_cannot_be_rescored_as_an_unrelated_scene(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"score_available": False, "scope": "recorded-fixed-views"}))
    with pytest.raises(ValueError, match="no ground truth or AP"):
        runner.rescore_result(path, "helsinki")
