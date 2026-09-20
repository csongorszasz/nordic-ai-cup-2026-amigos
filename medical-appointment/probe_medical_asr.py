"""Same-clip CPU comparison of generic and medical full-v3, with no ASR hints."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import asr
from benchmark import write_json
from benchmark_alignment import load_inputs
from prepare_medical_whisper import MODEL as MEDICAL_MODEL, REVISION as MEDICAL_REVISION
from probe_asr_hints import MODEL as TURBO_MODEL, REVISION as TURBO_REVISION, changes


GENERIC_MODEL = "Systran/faster-whisper-large-v3"
GENERIC_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"


def word_time_errors(transcript):
    errors = []
    if not transcript["words"]:
        return [{"reason": "No timestamped words in the selected speech clip."}]
    for word in transcript["words"]:
        if not (
            math.isfinite(word["start"]) and math.isfinite(word["end"])
            and 0 <= word["start"] <= word["end"] <= transcript["duration"]
        ):
            errors.append({**word, "reason": "Word time is not finite, ordered and within the audio."})
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--turbo-diagnostic", type=Path, required=True)
    parser.add_argument("--medical-preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/medical_asr_comparison"))
    args = parser.parse_args()
    _, requests, _, audio = load_inputs(args.baseline)
    preparation = json.loads(args.medical_preparation.read_text())
    if (
        not preparation["complete"] or preparation["model"] != MEDICAL_MODEL
        or preparation["revision"] != MEDICAL_REVISION or not preparation["alignment_heads_preserved"]
        or not preparation["smoke_words"] or preparation["quantization"] != "int8"
    ):
        raise ValueError("A verified medical checkpoint conversion is required.")
    medical_path = Path(preparation["converted_path"])
    for entry in preparation["files"]:
        path = medical_path / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Converted medical model file changed: {entry['path']}")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"Converted medical model hash changed: {entry['path']}")
    previous = json.loads((args.turbo_diagnostic / "summary.json").read_text())
    runtime = {name: importlib.metadata.version(name) for name in ("faster-whisper", "ctranslate2")}
    if (
        not previous["complete"] or previous["model"] != TURBO_MODEL or previous["revision"] != TURBO_REVISION
        or previous["actual_device"] != "cpu" or previous["runtime"] != runtime
    ):
        raise ValueError("The completed matched CPU turbo diagnostic is required.")
    tids = [case["transcript_id"] for case in previous["cases"]]
    if not tids or len(tids) != len(set(tids)) or any(tid not in audio for tid in tids):
        raise ValueError("Diagnostic cases are missing, repeated or not supplied training data.")
    turbo_outputs = {}
    for case in previous["cases"]:
        tid = case["transcript_id"]
        if hashlib.sha256(audio[tid]).hexdigest() != case["audio_sha256"]:
            raise ValueError("The turbo and full-v3 comparisons use different audio.")
        turbo_outputs[tid] = json.loads((args.turbo_diagnostic / "control" / f"{tid}.json").read_text())
        if turbo_outputs[tid].get("prompting") or word_time_errors(turbo_outputs[tid]):
            raise ValueError("The turbo comparison must use its valid unprompted control.")
    if any(preparation["runtime"][name] != runtime[name] for name in runtime):
        raise ValueError("The converted-model and comparison ASR runtimes differ.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Comparison output already exists; use a fresh run.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This recognition comparison must use the CPU-only runner.")
    import ctranslate2
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if threads < 1 or threads != previous["cpu_threads"]:
        raise ValueError("Use the same positive CPU thread allocation as the turbo diagnostic.")
    generic_path = snapshot_download(GENERIC_MODEL, revision=GENERIC_REVISION, local_files_only=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        if not (Path(generic_path) / name).is_file():
            raise FileNotFoundError(f"Generic control lacks required offline file: {name}")
    request_map = {entry["transcript_id"]: entry for entry in requests}
    outputs, runs, models = {}, {}, {}
    asr.DEVICE, asr.COMPUTE_TYPE, asr.LANGUAGE = "cpu", "int8", "en"
    asr.INITIAL_PROMPT = asr.HOTWORDS = ""
    for name, path, model_id, revision in (
        ("generic_full_v3", generic_path, GENERIC_MODEL, GENERIC_REVISION),
        ("medical_full_v3", str(medical_path), MEDICAL_MODEL, MEDICAL_REVISION),
    ):
        model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=threads, local_files_only=True)
        if model.model.device != "cpu" or model.model.compute_type != previous["actual_compute_type"]:
            raise RuntimeError("ASR model did not use the matched CPU compute type.")
        asr.MODEL_SIZE, asr._model = path, model
        asr.warm_up()
        outputs[name], runs[name] = {}, {}
        models[name] = {
            "model": model_id, "revision": revision, "path": path,
            "actual_device": model.model.device, "actual_compute_type": model.model.compute_type,
        }
        for tid in tids:
            ctranslate2.set_random_seed(previous["seed"])
            started = time.monotonic()
            transcript = asr.transcribe_bytes(
                audio[tid], request_map[tid]["audio_filename"], cache=False, force=True,
            )
            elapsed = time.monotonic() - started
            outputs[name][tid] = transcript
            runs[name][tid] = {
                "elapsed_s": elapsed, "words": len(transcript["words"]),
                "word_time_errors": word_time_errors(transcript),
            }
            write_json(args.output / name / f"{tid}.json", transcript)
            write_json(args.output / "progress.json", runs)
            print(f"{name} {tid}: {elapsed:.2f}s, {len(transcript['words'])} words", flush=True)
        asr._model = None
        del model
        gc.collect()
    terms = [term.strip().strip(".") for term in previous["glossary"].split(",")]
    cases = []
    for tid in tids:
        turbo = turbo_outputs[tid]
        cases.append({
            "transcript_id": tid, "audio_sha256": hashlib.sha256(audio[tid]).hexdigest(),
            "generic_vs_turbo": changes(turbo, outputs["generic_full_v3"][tid], terms),
            "medical_vs_generic": changes(outputs["generic_full_v3"][tid], outputs["medical_full_v3"][tid], terms),
            "medical_vs_turbo": changes(turbo, outputs["medical_full_v3"][tid], terms),
        })
    report = {
        "complete": True, "diagnostic_only": True, "not_a_wer_measurement": True,
        "end_to_end_score_measured": False, "gpu_acceptance": False,
        "models": models, "runs": runs, "cases": cases, "runtime": runtime,
        "cpu_threads": threads, "seed": previous["seed"], "prompting": {},
        "medical_preparation_sha256": hashlib.sha256(args.medical_preparation.read_bytes()).hexdigest(),
        "turbo_summary_sha256": hashlib.sha256((args.turbo_diagnostic / "summary.json").read_bytes()).hexdigest(),
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}), flush=True)
    return 0 if all(not item["word_time_errors"] for group in runs.values() for item in group.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
