"""Benchmark faster-whisper models locally: speed (RTF) and GPU memory.

    BENCH_MODELS="large-v3,large-v3-turbo,distil-large-v3,medium.en" \
    WHISPER_COMPUTE_TYPE=int8 python local/bench_asr.py

Uses the same three conversations across models so the numbers compare.
"""

import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import audio_duration_seconds, load_sample_audio  # noqa: E402

MODELS = os.environ.get(
    "BENCH_MODELS", "large-v3,large-v3-turbo,distil-large-v3,medium.en"
).split(",")
SAMPLES = [
    "conversation_sample_17.mp3",
    "conversation_sample_4.mp3",
    "conversation_sample_10.mp3",
]
COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")


def gpu_used() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "?"


def bench(model_name: str) -> None:
    import torch
    from faster_whisper import WhisperModel

    started = time.time()
    try:
        model = WhisperModel(model_name, device="cuda", compute_type=COMPUTE)
    except Exception as exc:
        print(f"  {model_name}: LOAD FAILED ({exc})")
        return
    load_s = time.time() - started

    total_audio = total_time = 0.0
    for name in SAMPLES:
        audio = load_sample_audio(name)
        duration = audio_duration_seconds(audio) or 0.0
        with tempfile.NamedTemporaryFile(suffix=".mp3") as handle:
            handle.write(audio)
            handle.flush()
            began = time.perf_counter()
            segments = list(
                model.transcribe(
                    handle.name,
                    language="en",
                    word_timestamps=True,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )[0]
            )
            elapsed = time.perf_counter() - began
        total_audio += duration
        total_time += elapsed
        print(f"    {name:<32} {duration:6.1f}s audio  {elapsed:6.1f}s  "
              f"{duration / elapsed if elapsed else 0:4.1f}x  ({len(segments)} segs)")

    rtf = total_audio / total_time if total_time else 0.0
    print(
        f"  {model_name:<22} load {load_s:4.1f}s  total {total_time:5.1f}s  "
        f"RTF {rtf:4.1f}x  gpu {gpu_used()}"
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    print(f"compute_type={COMPUTE}  samples={len(SAMPLES)}")
    for model_name in MODELS:
        model_name = model_name.strip()
        if model_name:
            bench(model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
