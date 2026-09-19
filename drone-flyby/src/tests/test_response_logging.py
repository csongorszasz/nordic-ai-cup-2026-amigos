import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from dtos import DroneFlybyPredictionDto, DroneFlybyPredictResponseDto
from offline import record_dataset
from tests.test_scaffolding import create_synthetic_request


def api_module(tmp_path, monkeypatch):
    def predict(request):
        return DroneFlybyPredictResponseDto(
            request_id=request.request_id, frame=request.frame,
            annotations=[DroneFlybyPredictionDto(
                object_id='tank', bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.5002,
            )],
            requested_view=None,
        )
    monkeypatch.setenv('DRONE_STACK', 'solution')
    monkeypatch.setenv('DRONE_LOG_RESPONSES', str(tmp_path / 'responses.jsonl'))
    monkeypatch.setitem(sys.modules, 'solution', SimpleNamespace(predict=predict))
    monkeypatch.setattr(record_dataset, 'recorder_from_env', lambda: None)
    spec = importlib.util.spec_from_file_location('api_logging_test', Path(__file__).resolve().parents[1] / 'api.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validated_response_log_preserves_identity_precision_and_scope(tmp_path, monkeypatch):
    api = api_module(tmp_path, monkeypatch)
    request = create_synthetic_request()
    response = api.predict_endpoint(request)
    entry = json.loads((tmp_path / 'responses.jsonl').read_text())
    assert entry['request_id'] == request.request_id
    assert entry['frame_index'] == request.frame_index
    assert entry['original_width'] == request.original_width
    assert entry['response_validated'] is True
    assert entry['evaluator_accepted'] is None
    assert entry['annotations'] == [a.model_dump(mode='json') for a in response.annotations]
    assert entry['annotations'][0]['confidence'] == 0.5002
    assert api.hello()['response_log']['written'] == 1


def test_logging_io_failure_does_not_erase_a_valid_prediction(tmp_path, monkeypatch, caplog):
    api = api_module(tmp_path, monkeypatch)
    api.RESPONSE_LOG = str(tmp_path)  # An existing directory cannot be opened as a log file.
    request = create_synthetic_request()
    response = api.predict_endpoint(request)
    assert response.request_id == request.request_id and response.annotations
    assert api.hello()['response_log']['errors'] == 1
    assert 'prediction is retained' in caplog.text


def test_invalid_response_is_not_written_as_a_valid_answer(tmp_path, monkeypatch):
    api = api_module(tmp_path, monkeypatch)
    def reject(response):
        raise ValueError('simulated invalid response')
    monkeypatch.setattr(api, 'validate_response', reject)
    with pytest.raises(ValueError, match='simulated invalid'):
        api.predict_endpoint(create_synthetic_request())
    assert not (tmp_path / 'responses.jsonl').exists()


def test_wrong_response_identity_is_rejected_before_logging(tmp_path, monkeypatch):
    api = api_module(tmp_path, monkeypatch)
    original = api.predict
    def wrong(request):
        return original(request).model_copy(update={'frame': request.frame + 1})
    monkeypatch.setattr(api, 'predict', wrong)
    with pytest.raises(ValueError, match='identity'):
        api.predict_endpoint(create_synthetic_request())
    assert not (tmp_path / 'responses.jsonl').exists()
