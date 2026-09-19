"""Timing-only Qwen alignment comparison with frozen decisions and word anchors."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path

from answerers.align import align_quote_matches
from answerers.base import normalize_answer
from answerers.boundaries import adjusted_span
from answerers.modernbert_data import load_rows
from answerers.retime import retime_words
from benchmark import paired_comparison, write_json
from check_alignment_tokens import MODEL, REVISION
from llm_probe import score_records
from train_quote_oof import validate_coverage
from utils import audio_filename_for_transcript, gold_evidence, load_sample_audio


logger = logging.getLogger(__name__)


def word_anchor(row, words):
    if not row["answer"]:
        return None
    anchor = row.get("word_range")
    if (
        not isinstance(anchor, list) or len(anchor) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in anchor)
        or not 0 <= anchor[0] <= anchor[1] < len(words)
    ):
        raise ValueError(f"Missing or invalid frozen word anchor: {row['question_id']}")
    matches = align_quote_matches(words, row.get("quote") or "")
    matched = next((match for match in matches if list(match[2:]) == anchor), None)
    if matched is None or any(abs(left - right) > 1e-6 for left, right in zip(matched[:2], row["span"])):
        raise ValueError(f"Frozen quote/span does not match its occurrence: {row['question_id']}")
    return anchor


def baseline_prediction(row, reason=None):
    return {
        **row,
        "span": adjusted_span(row["span"], (0.2, 0.0), row["duration"]) if row["answer"] else None,
        "alignment_reason": reason if row["answer"] else "base_no",
    }


def retime_prediction(row, original_words, aligned_words):
    fallback = baseline_prediction(row, "invalid_span_keep")
    anchor = word_anchor(row, original_words)
    if anchor is None:
        return fallback
    if [word["word"] for word in original_words] != [word["word"] for word in aligned_words]:
        raise ValueError("Retiming changed the frozen ASR word text/order.")
    span = [aligned_words[anchor[0]]["start"], aligned_words[anchor[1]]["end"]]
    valid, checked = normalize_answer(
        (True, span), duration=row["duration"], context=f"alignment {row['question_id']}",
    )
    if not valid:
        return fallback
    return {**row, "span": list(checked), "alignment_reason": "retimed"}


def load_inputs(directory):
    rows = json.loads((directory / "base_legacy_questions.json").read_text())
    requests = json.loads((directory / "base_legacy_conversations.json").read_text())
    truth = {row["question_id"]: row for row in load_rows()}
    if len(rows) != len(truth) or {row["question_id"] for row in rows} != truth.keys():
        raise ValueError("The complete unique frozen training-corpus baseline is required.")
    tids = {row["transcript_id"] for row in rows}
    if len(requests) != len(tids) or {entry["transcript_id"] for entry in requests} != tids:
        raise ValueError("The baseline conversation manifest is incomplete or repeated.")
    transcripts, audio = {}, {}
    for entry in requests:
        tid = entry["transcript_id"]
        transcript = json.loads((directory / "transcripts" / f"{tid}.json").read_text())
        if hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest() != entry["transcript_sha256"]:
            raise ValueError(f"Frozen transcript changed: {tid}")
        if not math.isfinite(transcript["duration"]) or not 0 < transcript["duration"] <= 300:
            raise ValueError(f"Audio exceeds the pinned aligner's supported duration: {tid}")
        if entry["audio_filename"] != audio_filename_for_transcript(tid):
            raise ValueError(f"Unexpected training audio filename: {tid}")
        audio[tid] = load_sample_audio(entry["audio_filename"])
        if hashlib.sha256(audio[tid]).hexdigest() != entry["audio_sha256"]:
            raise ValueError(f"Frozen audio changed: {tid}")
        transcripts[tid] = transcript
    for row in rows:
        reference = truth[row["question_id"]]
        gold = gold_evidence(reference)
        transcript = transcripts[row["transcript_id"]]
        if (
            any(row[key] != reference[key] for key in ("transcript_id", "question", "question_type"))
            or row["label"] != int(reference["label"])
            or row["gold"] != (list(gold) if gold is not None else None)
            or not isinstance(row["answer"], bool) or row["prediction"] != int(row["answer"])
            or row["duration"] != transcript["duration"]
        ):
            raise ValueError(f"Baseline question/reference mismatch: {row['question_id']}")
        valid, checked = normalize_answer(
            (row["answer"], row["span"]), duration=row["duration"], context=row["question_id"],
        )
        if valid != row["answer"] or (list(checked) if checked is not None else None) != row["span"]:
            raise ValueError(f"Frozen answer violates the evidence contract: {row['question_id']}")
        word_anchor(row, transcript["words"])
    return rows, requests, transcripts, audio


def align_audio(model, processor, audio_bytes, transcript):
    import torch
    from faster_whisper.audio import decode_audio

    sampling_rate = processor.feature_extractor.sampling_rate
    waveform = decode_audio(io.BytesIO(audio_bytes), sampling_rate=sampling_rate)
    duration = len(waveform) / sampling_rate
    if abs(duration - transcript["duration"]) > 1 / sampling_rate + 1e-9:
        raise ValueError("Decoded waveform and frozen transcript have different durations.")
    inputs, word_lists = processor.prepare_forced_aligner_inputs(
        audio=waveform, transcript="".join(word["word"] for word in transcript["words"]), language="English",
    )
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        logits = model(**inputs).logits
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Forced aligner produced non-finite logits.")
    batches = processor.decode_forced_alignment(
        logits=logits, input_ids=inputs["input_ids"], word_lists=word_lists,
        timestamp_token_id=model.config.timestamp_token_id,
    )
    if len(batches) != 1 or len(batches[0]) != len(word_lists[0]):
        raise ValueError("Forced alignment omitted or duplicated units.")
    return batches[0]


def alignment_gates(report):
    successful = report["conversation_failures"] == 0 and report["reasons"].get("retimed", 0) > 0
    return {
        "alignment_success": successful,
        "feasibility_passed": (
            successful and report["device"] == "cuda" and report["max_estimated_combined_s"] < 50
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Only the three longest frozen conversations.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="CPU uses float32 for quality diagnosis only, never a GPU latency gate.")
    parser.add_argument("--output", type=Path, default=Path("results/alignment_benchmark"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rows, requests, transcripts, audio = load_inputs(args.baseline)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Alignment output already contains a run; use a fresh directory.")
    if args.smoke:
        requests = sorted(requests, key=lambda entry: transcripts[entry["transcript_id"]]["duration"], reverse=True)[:3]
    selected = {entry["transcript_id"] for entry in requests}
    rows = [row for row in rows if row["transcript_id"] in selected]
    excluded = {tid for entry in requests for tid in entry["demonstration_tids"]}
    args.output.mkdir(parents=True, exist_ok=True)

    import resource
    import torch

    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("This timing experiment requires exactly its allocated GPU.")
        dtype = torch.bfloat16
    else:
        if torch.cuda.is_available():
            raise RuntimeError("CPU quality diagnostics require a CPU-only allocation with GPUs hidden.")
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        if threads < 1:
            raise ValueError("The allocated CPU thread count must be positive.")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        dtype = torch.float32
    from transformers import AutoModelForTokenClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, local_files_only=True,
    ).to(args.device).eval()
    if next(model.parameters()).dtype != dtype:
        raise RuntimeError("Forced-aligner weights did not load in the declared precision.")
    print(json.dumps({"device": args.device, "dtype": str(dtype), "cpu_threads": torch.get_num_threads()}), flush=True)
    warm = requests[0]["transcript_id"]
    align_audio(model, processor, audio[warm], transcripts[warm])
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    predictions, timings = [], []
    for entry in requests:
        tid = entry["transcript_id"]
        transcript = transcripts[tid]
        group = [row for row in rows if row["transcript_id"] == tid]
        began = time.monotonic()
        error = None
        try:
            units = align_audio(model, processor, audio[tid], transcript)
            write_json(args.output / "units" / f"{tid}.json", units)
            aligned_words = retime_words(transcript["words"], units, transcript["duration"])
            changed = [retime_prediction(row, transcript["words"], aligned_words) for row in group]
            write_json(args.output / "words" / f"{tid}.json", aligned_words)
        except Exception as exc:
            logger.exception("Alignment failed for %s; retaining every incumbent answer/span.", tid)
            error = f"{type(exc).__name__}: {exc}"
            changed = [baseline_prediction(row, "conversation_error_keep") for row in group]
            if args.device == "cuda":
                torch.cuda.empty_cache()
        elapsed = time.monotonic() - began
        validate_coverage(changed, group)
        predictions.extend(changed)
        timings.append({
            "transcript_id": tid, "alignment_s": elapsed, "error": error,
            "estimated_combined_s": entry["latency_s"] + elapsed,
        })
        write_json(args.output / "questions.json", predictions)
        write_json(args.output / "conversations.json", timings)
        print(f"{tid}: alignment={elapsed:.3f}s error={error}", flush=True)
    validate_coverage(predictions, rows)
    baseline = [baseline_prediction(row) for row in rows]
    safe_base = [row for row in baseline if row["transcript_id"] not in excluded]
    safe_pred = [row for row in predictions if row["transcript_id"] not in excluded]
    report = {
        "complete": True, "smoke_only": args.smoke, "questions": len(rows),
        "device": args.device, "dtype": str(dtype), "cpu_threads": torch.get_num_threads(),
        "quality_only": args.device == "cpu",
        "model": MODEL, "revision": REVISION, "timestamp_quantum_s": processor.timestamp_segment_time / 1000,
        "baseline": score_records(baseline), "aligned": score_records(predictions),
        "paired": paired_comparison(baseline, predictions),
        "demonstration_disjoint": {
            "baseline": score_records(safe_base), "aligned": score_records(safe_pred),
            "paired": paired_comparison(safe_base, safe_pred),
        },
        "excluded_demonstration_tids": sorted(excluded),
        "reasons": dict(Counter(row["alignment_reason"] for row in predictions)),
        "conversation_failures": sum(entry["error"] is not None for entry in timings),
        "max_alignment_s": max(entry["alignment_s"] for entry in timings),
        "max_estimated_combined_s": max(entry["estimated_combined_s"] for entry in timings),
        "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None,
        "peak_process_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "faster-whisper")},
        "latency_note": (
            "CPU float32 quality diagnostic; GPU numerical parity and co-resident HTTP latency unverified."
            if args.device == "cpu" else
            "Separate-run sum with only aligner loaded, not a co-resident HTTP gate."
        ),
        "word_text_and_anchors_frozen": True, "aligner_offsets": [0.0, 0.0],
    }
    report.update(alignment_gates(report))
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["alignment_success" if args.device == "cpu" else "feasibility_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
