"""B1: native-audio transcription-quality probe with Gemma 4.

Ask a multimodal Gemma 4 model to transcribe the consultation audio directly
(no Whisper) and compare the medical terms/units against the current ASR
transcripts — the known failure list: PAMEL/Pamol, ibumedin/Ibumetin,
panadil/Panodil, Activel/Activelle, Isomeprazole/Esomeprazole,
"long-term sugar value"/HbA1c, BP "over" vs slash, units.

    python llm_audio_probe.py --tids sample_20 sample_5 sample_19
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

MODEL = "google/gemma-4-e4b-it"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
TRANSCRIPTS = PROJECT_ROOT / "transcripts"

INSTRUCTION = (
    "Transcribe this doctor-patient consultation verbatim in English. "
    "Write every medication name, dose, unit and number exactly as spoken. "
    "Output only the transcript text, no commentary."
)

# terms to look for, and the Whisper variant they should replace
CHECKS = {
    "sample_20": [("Pamol", "PAMEL"), ("Ibumetin", "ibumedin")],
    "sample_5": [("Panodil", "panadil"), ("Activelle", "Activel"), ("Ibumetin", "ibumetin")],
    "sample_19": [("Activelle", "Activel"), ("Esomeprazole", "Isomeprazole")],
    "sample_55": [("HbA1c", "long-term sugar value")],
}


def whisper_text(tid: str) -> str:
    path = TRANSCRIPTS / f"conversation_{tid}.dc5ba020.json"
    if not path.exists():
        path = sorted(TRANSCRIPTS.glob(f"conversation_{tid}.*.json"))[0]
    return json.loads(path.read_text())["segments"]


def load_model(model_name: str, dtype: str = "float16"):
    processor = AutoProcessor.from_pretrained(model_name)
    torch_dtype = getattr(torch, dtype, torch.float16)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch_dtype)
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def transcribe(processor, model, device, tid: str, max_new_tokens: int) -> str:
    audio_bytes = (AUDIO_DIR / f"conversation_{tid}.mp3").read_bytes()
    waveform = decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(
        text=[text], audio=[waveform], sampling_rate=16000, return_tensors="pt"
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[0][encoded["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tids", nargs="+", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--out-dir", default="results/audio_probe")
    args = parser.parse_args()

    processor, model, device = load_model(args.model)
    print("model loaded on", device, flush=True)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for tid in args.tids:
        started = time.perf_counter()
        try:
            native = transcribe(processor, model, device, tid, args.max_new_tokens)
        except Exception as exc:
            print(f"{tid}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        elapsed = time.perf_counter() - started
        (out_dir / f"{tid}.txt").write_text(native)
        print(f"\n===== {tid}  ({elapsed:.1f}s, {len(native.split())} words) =====")
        print(native[:1200])
        for good, bad in CHECKS.get(tid, []):
            has_good = good.lower() in native.lower()
            has_bad = bad.lower() in native.lower()
            print(f"   check {good!r}: native={has_good}  whisper_variant={bad!r}={has_bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
