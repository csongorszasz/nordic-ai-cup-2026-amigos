"""Paired CPU ASR vocabulary diagnostic; not WER or an end-to-end score."""

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

import asr
from benchmark import write_json
from benchmark_alignment import load_inputs


MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
NEGATIONS = re.compile(r"\b(?:no|not|never|without|denies|don't|doesn't|isn't|aren't|haven't|wasn't)\b", re.I)
NUMBERS = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def text_of(transcript):
    return "".join(word["word"] for word in transcript["words"])


def changes(before, after, terms):
    left, right = text_of(before), text_of(after)
    before_words = [word["word"].strip() for word in before["words"]]
    after_words = [word["word"].strip() for word in after["words"]]
    edits = []
    for tag, a, b, c, d in difflib.SequenceMatcher(
        a=before_words, b=after_words, autojunk=False,
    ).get_opcodes():
        if tag != "equal":
            edits.append({"kind": tag, "before": before_words[a:b], "after": after_words[c:d]})
    return {
        "edit_blocks": edits,
        "number_tokens_before": NUMBERS.findall(left),
        "number_tokens_after": NUMBERS.findall(right),
        "negation_tokens_before": [word.lower() for word in NEGATIONS.findall(left)],
        "negation_tokens_after": [word.lower() for word in NEGATIONS.findall(right)],
        "term_counts": {
            term: {
                "before": len(re.findall(r"\b" + re.escape(term) + r"\b", left, re.I)),
                "after": len(re.findall(r"\b" + re.escape(term) + r"\b", right, re.I)),
            }
            for term in terms
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--transcript-ids", nargs="+", required=True)
    parser.add_argument("--glossary", required=True, help="Fixed, short vocabulary; no doses or question assertions.")
    parser.add_argument("--output", type=Path, default=Path("results/asr_hints"))
    args = parser.parse_args()
    _, requests, transcripts, audio = load_inputs(args.baseline)
    if len(args.transcript_ids) != len(set(args.transcript_ids)) or any(
        tid not in transcripts for tid in args.transcript_ids
    ):
        raise ValueError("Choose unique supplied training conversations.")
    glossary = args.glossary.strip()
    terms = [part.strip().strip(".") for part in glossary.split(",") if part.strip().strip(".")]
    if not terms:
        raise ValueError("A non-empty vocabulary is required.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("ASR diagnostic output already exists; use a fresh run.")
    args.output.mkdir(parents=True, exist_ok=True)
    import ctranslate2
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This preliminary ASR diagnostic must use the CPU-only runner.")
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if threads < 1:
        raise ValueError("The allocated CPU thread count must be positive.")
    model_path = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    model = WhisperModel(
        model_path, device="cpu", compute_type="int8",
        cpu_threads=threads, local_files_only=True,
    )
    if model.model.device != "cpu":
        raise RuntimeError("ASR diagnostic unexpectedly selected a GPU.")
    asr.MODEL_SIZE, asr.DEVICE, asr.COMPUTE_TYPE, asr.LANGUAGE = model_path, "cpu", "int8", "en"
    asr._model = model
    asr.INITIAL_PROMPT = asr.HOTWORDS = ""
    asr.warm_up()
    request_map = {entry["transcript_id"]: entry for entry in requests}
    report = {
        "diagnostic_only": True, "not_a_wer_measurement": True,
        "end_to_end_score_measured": False, "gpu_acceptance": False,
        "model": MODEL, "revision": REVISION, "model_path": model_path,
        "actual_device": model.model.device, "actual_compute_type": model.model.compute_type,
        "cpu_threads": threads, "seed": 13, "glossary": glossary,
        "dynamic_question_prompting": False,
        "runtime": {name: importlib.metadata.version(name) for name in ("faster-whisper", "ctranslate2")},
        "cases": [],
    }
    for tid in args.transcript_ids:
        outputs, runs = {}, {}
        for mode in ("control", "initial_prompt", "hotwords"):
            asr.INITIAL_PROMPT = glossary if mode == "initial_prompt" else ""
            asr.HOTWORDS = glossary if mode == "hotwords" else ""
            ctranslate2.set_random_seed(13)
            started = time.monotonic()
            transcript = asr.transcribe_bytes(
                audio[tid], request_map[tid]["audio_filename"], cache=False, force=True,
            )
            elapsed = time.monotonic() - started
            outputs[mode] = transcript
            runs[mode] = {
                "elapsed_s": elapsed, "config_hash": asr.config_hash(),
                "words": len(transcript["words"]), "segments": len(transcript["segments"]),
                "temperatures": sorted({
                    segment["temperature"] for segment in transcript["segments"]
                    if segment["temperature"] is not None
                }),
            }
            write_json(args.output / mode / f"{tid}.json", transcript)
            print(f"{tid} {mode}: {elapsed:.2f}s, {len(transcript['words'])} words", flush=True)
        report["cases"].append({
            "transcript_id": tid, "audio_sha256": hashlib.sha256(audio[tid]).hexdigest(),
            "runs": runs,
            "cpu_control_vs_cached_gpu": changes(transcripts[tid], outputs["control"], terms),
            "initial_prompt_vs_cpu_control": changes(outputs["control"], outputs["initial_prompt"], terms),
            "hotwords_vs_cpu_control": changes(outputs["control"], outputs["hotwords"], terms),
        })
        write_json(args.output / "summary.json", report)
    report["complete"] = True
    write_json(args.output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
