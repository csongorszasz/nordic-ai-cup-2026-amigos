"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI

from contextlib import asynccontextmanager

from config import DEFAULT_CONFIG
from core import build_pipeline
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = build_pipeline(DEFAULT_CONFIG)
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models and kernels on server startup
    pipeline.warmup()
    yield


app = FastAPI(lifespan=lifespan)


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
def predict_endpoint(request: DroneFlybyPredictRequestDto):
    """Answer one frame."""
    response = pipeline.handle_request(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
