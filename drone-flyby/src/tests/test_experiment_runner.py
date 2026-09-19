import argparse
import json

import pytest

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
