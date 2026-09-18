# GPU image for the served LLM answerer (ASR + Gemma), for any host with an
# NVIDIA GPU and the container toolkit:
#
#   docker build -t medical-appointment .
#   docker run --gpus all -p 9054:9054 \
#       -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#       -e MEDAPP_LLM_REVISION=<hash from models/llm_revisions.txt> \
#       medical-appointment
#
# Weights come from the mounted Hugging Face cache (pre-downloaded; the
# container runs offline and never fetches at request time). The few-shot
# examples need data/, annotations/ and transcripts/ in the build context;
# with MEDAPP_STRICT_STARTUP=1 the container refuses to start without them.
#
# cuDNN 9 is what current ctranslate2 (faster-whisper) is built against.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    MEDAPP_ANSWERER=llm \
    MEDAPP_LLM_MODEL=google/gemma-4-e4b-it \
    MEDAPP_LLM_DTYPE=float16 \
    WHISPER_MODEL=large-v3-turbo \
    WHISPER_COMPUTE_TYPE=int8 \
    MEDAPP_STRICT_STARTUP=1 \
    MEDAPP_ASR_CACHE=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Match the validated IDUN env where a lock exists (idun/setup_llm.sh writes it);
# only the packages that decide model behaviour are constrained.
RUN if [ -f idun/serving-env.lock.txt ]; then \
        grep -iE '^(torch|transformers|tokenizers|accelerate|faster-whisper|ctranslate2|huggingface-hub|numpy)==' \
            idun/serving-env.lock.txt > /tmp/constraints.txt; \
    else : > /tmp/constraints.txt; fi \
    && pip3 install --no-cache-dir -c /tmp/constraints.txt torch \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir -c /tmp/constraints.txt \
        -r requirements.txt -r requirements-model.txt faster-whisper numpy accelerate

EXPOSE 9054
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -sf http://localhost:9054/ready || exit 1

CMD ["python3", "api.py"]
