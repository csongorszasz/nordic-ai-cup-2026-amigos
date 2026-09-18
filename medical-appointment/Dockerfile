FROM python:3.11-slim

WORKDIR /app

# libgomp is needed by torch/ctranslate2.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-model.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-model.txt \
    && pip install --no-cache-dir faster-whisper \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The whole solution (the old image copied only four files).
COPY asr.py windows.py questions.py guards.py answer.py capture.py example.py \
     api.py dtos.py utils.py ./
COPY verifier ./verifier

# This image is CPU-only: it validates the wiring but is far too slow for the
# 60 s budget. For GPU serving use an nvidia/cuda base (cuDNN runtime) and
# `torch --index-url https://download.pytorch.org/whl/cu121`, then run with
# `--gpus all`. Model weights are pre-downloaded via `local/setup_env.sh`
# (mounted or baked in) — the container must not fetch them at request time.
EXPOSE 9054

CMD ["python", "api.py"]
