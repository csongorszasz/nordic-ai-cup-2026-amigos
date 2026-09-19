"""Offline, run-local medical Whisper conversion and word-timestamp feasibility."""

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path

from benchmark import write_json
from utils import load_sample_audio


MODEL = "Na0s/Medical-Whisper-Large-v3"
REVISION = "9943ad3338e2ffdcdadb193d9e2abc9feeded448"


def verify_alignment_heads(source, converted):
    heads = source.get("alignment_heads")
    if not heads or converted.get("alignment_heads") != heads:
        raise ValueError("Conversion did not preserve the checkpoint's alignment heads.")
    return len(heads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, default=Path("models/medical_whisper_ct2"))
    parser.add_argument("--output", type=Path, default=Path("results/medical_whisper_preparation.json"))
    args = parser.parse_args()
    manifest = json.loads(args.cache_manifest.read_text())
    if not manifest["complete"] or manifest["model"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("The exact completed medical checkpoint cache is required.")
    source = Path(manifest["path"])
    for entry in manifest["files"]:
        path = source / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Cached checkpoint file changed: {entry['path']}")
    if args.model_output.exists() or args.output.exists():
        raise FileExistsError("Use a fresh conversion destination and report.")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Conversion must run offline in the isolated CPU runner.")
    import resource
    import torch
    from ctranslate2.converters import TransformersConverter
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from transformers import AutoTokenizer

    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if threads < 1:
        raise ValueError("Allocated CPU count must be positive.")
    torch.set_num_threads(threads)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    converter = TransformersConverter(
        str(source), copy_files=["preprocessor_config.json"],
        load_as_float16=False, low_cpu_mem_usage=True, trust_remote_code=False,
    )
    converter.convert(str(args.model_output), quantization="int8")
    del converter
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise RuntimeError("An offline fast tokenizer is required for faster-whisper.")
    tokenizer.save_pretrained(args.model_output)
    if not (args.model_output / "tokenizer.json").is_file():
        raise RuntimeError("Converted model lacks the tokenizer needed for offline inference.")
    generation = json.loads((source / "generation_config.json").read_text())
    converted = json.loads((args.model_output / "config.json").read_text())
    heads = verify_alignment_heads(generation, converted)
    audio = load_sample_audio("conversation_sample_5.mp3")
    waveform = decode_audio(io.BytesIO(audio), sampling_rate=16000)[:12 * 16000]
    model = WhisperModel(
        str(args.model_output), device="cpu", compute_type="int8", cpu_threads=threads,
        local_files_only=True,
    )
    segments, info = model.transcribe(
        waveform, language="en", word_timestamps=True,
        vad_filter=True, condition_on_previous_text=False,
    )
    segments = list(segments)
    words = [word for segment in segments for word in segment.words or []]
    if not words or any(
        not (math.isfinite(word.start) and math.isfinite(word.end)
             and 0 <= word.start <= word.end <= len(waveform) / 16000)
        for word in words
    ):
        raise RuntimeError("Converted medical Whisper failed the word-timestamp smoke.")
    files = []
    for path in sorted(args.model_output.iterdir()):
        if path.is_file():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest})
    report = {
        "complete": True, "feasibility_only": True, "model": MODEL, "revision": REVISION,
        "converted_path": str(args.model_output.resolve()), "quantization": "int8",
        "source_load_as_float16": False, "alignment_heads_preserved": heads,
        "actual_device": model.model.device, "actual_compute_type": model.model.compute_type,
        "cpu_threads": threads, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
        "smoke_audio_sha256": hashlib.sha256(audio).hexdigest(),
        "smoke_start_s": 0.0, "smoke_end_s": len(waveform) / 16000,
        "smoke_words": len(words), "smoke_text": "".join(word.word for word in words),
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "ctranslate2", "faster-whisper")},
        "source_cache_manifest_sha256": hashlib.sha256(args.cache_manifest.read_bytes()).hexdigest(),
        "files": files, "serving_environment_modified": False,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
