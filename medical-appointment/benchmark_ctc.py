"""Conditional acoustic timing probe with frozen decisions and source occurrences."""

import argparse
import hashlib
import io
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path

from answerers.acoustic_boundaries import (
    MODEL, RECIPE, REVISION, AcousticSkip, ConditionalCTC, load_number_renderer,
)
from audit_localization import exact_score, validate_qualified
from benchmark import paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs, word_anchor
from probe_source_localization import verify_model_cache
from train_quote_oof import validate_coverage
from utils import temporal_iou


logger = logging.getLogger(__name__)


def decode_waveform(audio):
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(audio), sampling_rate=RECIPE["sampling_rate"])


def validate_result(row, result):
    baseline = baseline_prediction(row)["span"]
    candidates = result.get("candidate_spans")
    if (
        not row["answer"] or result.get("source_word_range") != row["word_range"]
        or not isinstance(candidates, list) or not 1 <= len(candidates) <= RECIPE["max_endpoint_positions"] ** 2 + 1
        or candidates[0] != baseline or result.get("span") not in candidates
    ):
        raise ValueError("Acoustic proposals changed the occurrence, lost the incumbent, or exceeded their budget.")
    seen = set()
    for span in candidates:
        if (
            not isinstance(span, (list, tuple)) or len(span) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in span)
            or not 0 <= span[0] < span[1] <= row["duration"] or tuple(span) in seen
        ):
            raise ValueError("Acoustic proposals must be distinct finite intervals within the audio.")
        seen.add(tuple(span))


def score_subset(rows, predictions):
    if not rows:
        return None
    baseline = [baseline_prediction(row) for row in rows]
    selected = {row["question_id"]: row for row in predictions}
    predicted = [selected[row["question_id"]] for row in rows]
    return {
        "baseline": exact_score(baseline), "candidate": exact_score(predicted),
        "paired": paired_comparison(baseline, predicted),
    }


def run_benchmark(rows, requests, transcripts, audio, aligner, output, *, smoke=False, decoder=decode_waveform):
    import resource

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Conditional CTC diagnostics require an isolated CPU allocation.")
    excluded = sorted({tid for request in requests for tid in request["demonstration_tids"]})
    selected_requests = (
        sorted(requests, key=lambda entry: (-transcripts[entry["transcript_id"]]["duration"], entry["transcript_id"]))[:3]
        if smoke else requests
    )
    selected_tids = {entry["transcript_id"] for entry in selected_requests}
    expected = [row for row in rows if row["transcript_id"] in selected_tids]
    if not expected:
        raise ValueError("No matched questions are available for the acoustic benchmark.")
    load_started = time.monotonic()
    aligner.load()
    load_s = time.monotonic() - load_started
    predictions, traces, conversations, failures = [], [], [], []
    for entry in selected_requests:
        tid = entry["transcript_id"]
        group = [row for row in expected if row["transcript_id"] == tid]
        transcript = transcripts[tid]
        began = time.monotonic()
        waveform, decode_error = None, None
        try:
            waveform = decoder(audio[tid])
            if abs(len(waveform) / RECIPE["sampling_rate"] - transcript["duration"]) > 1 / RECIPE["sampling_rate"] + 1e-9:
                raise ValueError("The decoded sample clock differs from the frozen transcript.")
        except (RuntimeError, ValueError, OSError, MemoryError) as exc:
            logger.exception("Audio preparation failed for %s; retaining every incumbent result.", tid)
            decode_error = f"{type(exc).__name__}: {exc}"
            failures.append({"transcript_id": tid, "stage": "decode", "error": decode_error})
        for row in group:
            started = time.monotonic()
            prediction = {**baseline_prediction(row), "ctc_reason": "base_no" if not row["answer"] else "decode_error_keep"}
            trace = {
                "question_id": row["question_id"], "transcript_id": tid,
                "source_word_range": row.get("word_range"),
                "candidate_spans": [prediction["span"]] if row["answer"] else [],
                "error": decode_error if row["answer"] else None,
            }
            if row["answer"] and decode_error is None:
                try:
                    anchor = word_anchor(row, transcript["words"])
                    result = aligner.align(waveform, transcript["words"], anchor, prediction["span"])
                    validate_result(row, result)
                    prediction.update({"span": result["span"], "ctc_reason": "aligned"})
                    trace.update(result)
                except AcousticSkip as exc:
                    logger.warning("CTC source ineligible for %s; keeping incumbent: %s", row["question_id"], exc)
                    prediction["ctc_reason"] = "ineligible_keep"
                    trace["error"] = f"{type(exc).__name__}: {exc}"
                except (RuntimeError, ValueError, OSError, MemoryError) as exc:
                    logger.exception("CTC alignment failed for %s; keeping incumbent.", row["question_id"])
                    prediction["ctc_reason"] = "alignment_error_keep"
                    trace["error"] = f"{type(exc).__name__}: {exc}"
                    failures.append({"question_id": row["question_id"], "stage": "alignment", "error": trace["error"]})
            trace.update({"reason": prediction["ctc_reason"], "elapsed_s": time.monotonic() - started})
            predictions.append(prediction)
            traces.append(trace)
        elapsed = time.monotonic() - began
        conversations.append({
            "transcript_id": tid, "alignment_s": elapsed, "decode_error": decode_error,
            "estimated_combined_s": entry["latency_s"] + elapsed,
        })
        write_json(output / "questions.json", predictions)
        write_json(output / "proposals.json", traces)
        write_json(output / "conversations.json", conversations)
        print(json.dumps(conversations[-1]), flush=True)
    validate_coverage(predictions, expected)
    reasons = dict(Counter(row["ctc_reason"] for row in predictions))
    trace_by_id = {entry["question_id"]: entry for entry in traces}
    positives = [row for row in expected if row["label"] == 1]
    if not positives:
        raise ValueError("The matched acoustic benchmark contains no reference positives.")
    oracle = sum(
        max((temporal_iou(row["gold"], span) for span in trace_by_id[row["question_id"]]["candidate_spans"]), default=0.0)
        for row in positives
    ) / len(positives)
    report = {
        "complete": True, "full_corpus": not smoke, "smoke_only": smoke,
        "device": "cpu", "quality_only": True, "deployment_qualified": False,
        "serving_artifact_written": False, "word_text_and_anchors_frozen": True,
        "model": MODEL, "revision": REVISION, "recipe": RECIPE, "runtime": aligner.runtime,
        "load_s": load_s, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "questions": len(expected), "positives": len(positives), "reasons": reasons,
        "operational_failures": failures,
        "alignment_feasibility_passed": reasons.get("aligned", 0) > 0 and not failures,
        "all_questions": score_subset(expected, predictions),
        "demonstration_disjoint": score_subset([row for row in expected if row["transcript_id"] not in excluded], predictions),
        "excluded_demonstration_tids": excluded,
        "candidate_oracle_frozen_decision_mean_tiou": oracle,
        "max_alignment_s": max(entry["alignment_s"] for entry in conversations),
        "max_estimated_combined_s": max(entry["estimated_combined_s"] for entry in conversations),
        "limitations": [
            "All decisions, quote occurrences, references, and score denominators remain unchanged.",
            "Unsupported spoken forms or crop budgets explicitly retain the incumbent; their losses are not excluded.",
            "Candidate endpoint activations are not a calibrated posterior over reference boundaries.",
            "The proposal oracle uses gold only for diagnostics and is not a prediction or serving selector.",
            "CPU component timing and separate-run sums are not co-resident uncached HTTP acceptance.",
            "This pretrained acoustic trial does not establish near-perfect localization or hidden-set improvement.",
        ],
    }
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["alignment_feasibility_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--qualified", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--normalizer-manifest", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/ctc_boundaries"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh conditional acoustic output directory.")
    rows, requests, transcripts, audio = load_inputs(args.baseline)
    qualified = json.loads((args.qualified / "questions.json").read_text())
    qualified_summary = json.loads((args.qualified / "summary.json").read_text())
    validate_qualified([baseline_prediction(row) for row in rows], qualified, qualified_summary)
    model_proof = verify_model_cache(args.cache_manifest, model=MODEL, revision=REVISION)
    renderer = load_number_renderer(args.normalizer_manifest)
    write_json(args.output / "provenance.json", {
        "recipe": RECIPE, "model_cache": model_proof, "job_id": os.environ.get("SLURM_JOB_ID"),
        "normalizer_manifest_sha256": hashlib.sha256(args.normalizer_manifest.read_bytes()).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.baseline / "base_legacy_questions.json", args.baseline / "base_legacy_conversations.json",
                args.qualified / "questions.json", args.qualified / "summary.json",
            )
        },
    })
    return run_benchmark(
        rows, requests, transcripts, audio, ConditionalCTC(renderer), args.output, smoke=args.smoke,
    )


if __name__ == "__main__":
    raise SystemExit(main())
