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
import logging
import os
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
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models and kernels on server startup (solution.py warms up on import).
    if pipeline is not None:
        pipeline.warmup()
    yield


app = FastAPI(lifespan=lifespan)


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
    if recorder is not None:
        if not recorder.enabled or getattr(recorder, "session_dir", None) is None:
            recorder.start(request.sequence_id)
        elif recorder.session_dir.name != request.sequence_id:
            recorder.start(request.sequence_id)
        recorder.record_frame(request)

    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'stack': STACK,
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
