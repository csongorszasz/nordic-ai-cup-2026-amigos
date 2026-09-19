"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import os
import time

import uvicorn
from fastapi import FastAPI

from contextlib import asynccontextmanager

from config import DEFAULT_CONFIG
from core import build_pipeline
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from offline.record_dataset import recorder_from_env
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = build_pipeline(DEFAULT_CONFIG)
recorder = recorder_from_env()
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models and kernels on server startup
    pipeline.warmup()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    yield
    # Drain any pending recording jobs before the process exits.
    if recorder is not None:
        try:
            recorder.shutdown()
        except Exception:
            logger.exception("Failed to flush the validation recorder")


app = FastAPI(lifespan=lifespan)


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    t_start = time.perf_counter()

    if recorder is not None:
        # Recording must never break a frame: a dropped artifact is cheap, a
        # failed response is not.
        try:
            if (
                not recorder.enabled
                or recorder.session_dir is None
                or recorder.session_dir.name != request.sequence_id
            ):
                recorder.start(request.sequence_id)
            recorder.record_frame(request)
        except Exception:
            logger.exception("Failed to record the incoming validation frame")

    response = pipeline.handle_request(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    if recorder is not None:
        try:
            recorder.record_response(request, response, (time.perf_counter() - t_start) * 1000)
        except Exception:
            logger.exception("Failed to record the validation response")

    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/stats')
def stats():
    """Runtime counters used by the local benchmark harness."""
    peak_vram_mb = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
        'device': DEFAULT_CONFIG.DEVICE,
        'detector_type': DEFAULT_CONFIG.DETECTOR_TYPE,
        'tracker_type': DEFAULT_CONFIG.TRACKER_TYPE,
        'policy_type': DEFAULT_CONFIG.POLICY_TYPE,
        'peak_vram_mb': peak_vram_mb,
        'inference_shape': getattr(pipeline.detector, 'last_input_shape', None),
        'inference_dtype': getattr(pipeline.detector, 'last_input_dtype', None),
        'component_inference_shapes': getattr(pipeline.detector, 'component_inference_shapes', None),
        'component_inference_dtypes': getattr(pipeline.detector, 'component_inference_dtypes', None),
        'inference_image_size': DEFAULT_CONFIG.INFERENCE_IMAGE_SIZE,
        'inference_image_sizes': getattr(pipeline.detector, 'image_sizes', None),
        'inference_rect': DEFAULT_CONFIG.INFERENCE_RECT,
        'inference_half': DEFAULT_CONFIG.INFERENCE_HALF,
        'last_timings': pipeline.last_timings,
        'run_nonce': os.getenv('DRONE_FLYBY_RUN_NONCE'),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)
