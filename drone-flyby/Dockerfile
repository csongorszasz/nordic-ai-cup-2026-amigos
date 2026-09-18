# GPU-ready image for the drone-flyby server.
#
# The default detector is a task-finetuned YOLO11s model. A CPU-only image
# cannot meet the frame cadence, and an image without the weights must not
# silently fall back to a zero-scoring detector, so both are baked in here.
#
# Build:
#   docker build -t drone-flyby .
# Run (requires the NVIDIA container toolkit):
#   docker run --gpus all -p 9053:9053 drone-flyby
#
# For a CPU-only deployment, override the base image and the device:
#   docker build --build-arg BASE_IMAGE=python:3.11-slim -t drone-flyby-cpu .
#   docker run -e DRONE_FLYBY_DEVICE=cpu -p 9053:9053 drone-flyby-cpu

ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Server code plus the finetuned weights. The sample data under data/ is only
# needed by local_evaluator.py and is not required to serve requests.
COPY src/ ./
COPY weights/ ./weights/

ENV DRONE_FLYBY_DETECTOR_TYPE=yolo_standard \
    DRONE_FLYBY_YOLO_WEIGHTS_PATH=/app/weights/yolo11s_drone_flyby.pt

EXPOSE 9053

CMD ["python", "api.py"]
