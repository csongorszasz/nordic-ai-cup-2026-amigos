"""The endpoint the evaluation service calls.

You should not need to change much in here. The transport stays as the organisers
wrote it; the answer comes from one of two stacks, chosen with ``DRONE_STACK``:

``solution`` (default)
    ``solution.predict`` - fine-tuned YOLO + object memory + lag-safe camera planner.
    This is the stack that ran the live validation (0 refused camera moves).
``pipeline``
    ``core.build_pipeline`` - modular detector/tracker/camera-policy orchestrator.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import json
import logging
import os
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager

from config import DEFAULT_CONFIG
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from offline.record_dataset import recorder_from_env
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STACK = os.environ.get('DRONE_STACK', 'solution').strip().lower()
if STACK == 'pipeline':
    from core import build_pipeline

    pipeline = build_pipeline(DEFAULT_CONFIG)
    predict = pipeline.handle_request
else:
    from solution import predict

    pipeline = None

recorder = recorder_from_env()
# DRONE_LOG_RESPONSES=<file>: one JSON line per answered frame (view, boxes, time taken), no
# image, so a live run can be scored against our own labels (training/score_live_log.py).
RESPONSE_LOG = os.environ.get('DRONE_LOG_RESPONSES')
_response_log_lock = threading.Lock()
_response_log_counts = {'written': 0, 'errors': 0}
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models and kernels on server startup (solution.py warms up on import).
    if pipeline is not None:
        pipeline.warmup()
    yield


app = FastAPI(lifespan=lifespan)


def log_response(request, response, predict_ms, server_before_log_ms):
    """Optional diagnostics must not turn a valid prediction into an HTTP error."""
    if not RESPONSE_LOG:
        return
    entry = {
        'schema_version': 2, 'at': time.time(), 'ms': round(predict_ms),
        'predict_ms': predict_ms, 'server_before_log_ms': server_before_log_ms,
        'latency_scope': 'server time before logging/serialization/network',
        'sequence_id': request.sequence_id, 'request_id': request.request_id,
        'frame': request.frame, 'frame_index': request.frame_index,
        'original_width': request.original_width, 'original_height': request.original_height,
        'level': request.view.resolution_level, 'region': request.view.source_region_xyxy,
        'annotations': [annotation.model_dump(mode='json') for annotation in response.annotations],
        'requested_view': response.requested_view.model_dump(mode='json') if response.requested_view else None,
        'response_validated': True, 'evaluator_accepted': None,
    }
    try:
        line = json.dumps(entry, allow_nan=False) + '\n'
        with _response_log_lock:
            with open(RESPONSE_LOG, 'a', encoding='utf-8') as handle:
                handle.write(line)
            _response_log_counts['written'] += 1
    except (OSError, ValueError, TypeError):
        with _response_log_lock:
            _response_log_counts['errors'] += 1
        logger.exception('Failed to append the optional response log; prediction is retained')


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    # A 422 means the body did not match the request DTO. Log which fields and which
    # top-level keys arrived (never the image), e.g. a URL pasted into another use case's form.
    try:
        body = await request.json()
        keys = sorted(body.keys()) if isinstance(body, dict) else type(body).__name__
    except Exception:
        keys = 'unparseable body'
    logger.error('422 from %s: keys=%s errors=%s', request.client.host if request.client else '?',
                 keys, [(e.get('loc'), e.get('type')) for e in exc.errors()][:10])
    return JSONResponse(status_code=422, content={'detail': jsonable_encoder(exc.errors())})


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    received = time.monotonic()
    if recorder is not None:
        if not recorder.enabled or getattr(recorder, "session_dir", None) is None:
            recorder.start(request.sequence_id)
        elif recorder.session_dir.name != request.sequence_id:
            recorder.start(request.sequence_id)
        recorder.record_frame(request)

    took = time.monotonic()
    response = predict(request)
    predict_ms = (time.monotonic() - took) * 1000

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)
    if response.request_id != request.request_id or response.frame != request.frame:
        raise ValueError('Prediction identity does not match its request')
    log_response(request, response, predict_ms, (time.monotonic() - received) * 1000)

    return response


@app.get('/api')
def hello():
    with _response_log_lock:
        log_status = {'enabled': bool(RESPONSE_LOG), **_response_log_counts}
    return {
        'service': 'drone-flyby-usecase',
        'stack': STACK,
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
        'response_log': log_status,
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
