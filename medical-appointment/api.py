"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9054/predict``
rather than just the host.
"""

import datetime
import logging
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from example import READINESS, predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9054

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=ASRQuestionResponseDto)
def predict_endpoint(request: ASRQuestionRequestDto):
    """Answer every question about one conversation."""
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently score every
    # question about this conversation wrong.
    validate_response(response, expected_count=len(request.questions))

    return response


@app.get('/api')
def hello():
    return {
        'service': 'medical-appointment-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/ready')
def ready():
    """Readiness: 200 only once the models, few-shot and prior are loaded.

    ``/`` answers as soon as the process is up; poll this one before handing
    out the URL, so a failed warm-up is found before the attempt, not in it.
    """
    return JSONResponse(READINESS, status_code=200 if READINESS.get('ready') else 503)


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
