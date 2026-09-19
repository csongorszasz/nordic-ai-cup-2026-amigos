import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dtos import DroneFlybyPredictionDto, DroneFlybyPredictResponseDto, IMAGE_HEIGHT, IMAGE_WIDTH
import local_evaluator
from training import run_policy


def setup_replay(tmp_path, monkeypatch, *, loaded=True, actual_camera='sweep'):
    def predict(request):
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=[DroneFlybyPredictionDto(
                object_id='tank', confidence=0.5002,
                bbox=[10.49 / IMAGE_WIDTH, 10.49 / IMAGE_HEIGHT, 12.49 / IMAGE_WIDTH, 12.49 / IMAGE_HEIGHT],
            )],
            requested_view=None,
        )
    solution = SimpleNamespace(
        _model=object() if loaded else None, MODEL_PATH=Path('fixture/weights/model.pt'),
        CAMERA_POLICY=actual_camera, run_detector=lambda image, region: [], predict=predict, _states={},
    )
    monkeypatch.setitem(sys.modules, 'solution', solution)
    monkeypatch.setattr(run_policy, 'OUT', tmp_path)
    monkeypatch.setattr(run_policy, 'scene_frames', lambda scene: {
        1: lambda: np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
    })
    monkeypatch.setattr(sys, 'argv', ['run_policy', '--scene', 'fixture', '--camera', 'sweep', '--name', 'test'])
    return solution


def test_policy_scores_and_saves_full_precision_predictions(tmp_path, monkeypatch):
    setup_replay(tmp_path, monkeypatch)
    captured = {}
    def scorer(scene, predictions, truth=None):
        captured.update(predictions)
        return 0.123456789, {'tank': 0.123456789}
    monkeypatch.setattr(local_evaluator, 'score', scorer)
    run_policy.main()
    assert captured[1][0]['confidence'] == 0.5002
    assert captured[1][0]['bbox'][0] == pytest.approx(10.49)
    trace = json.loads((tmp_path / 'test.json').read_text())
    assert trace['map50'] == 0.123456789
    assert trace['steps'][0]['reported'][0]['conf'] == 0.5002
    assert trace['steps'][0]['reported'][0]['bbox'][0] == pytest.approx(10.49)


def test_policy_refuses_to_score_a_missing_detector_as_zero(tmp_path, monkeypatch):
    setup_replay(tmp_path, monkeypatch, loaded=False)
    with pytest.raises(RuntimeError, match='No detector loaded'):
        run_policy.main()
    assert not (tmp_path / 'test.json').exists()


def test_policy_uses_actual_camera_setting_for_record_mode_exception(tmp_path, monkeypatch):
    setup_replay(tmp_path, monkeypatch, loaded=False, actual_camera='record')
    monkeypatch.setattr(local_evaluator, 'score', lambda *args: (0.0, {'tank': 0.0}))
    run_policy.main()
    assert json.loads((tmp_path / 'test.json').read_text())['camera'] == 'record'
