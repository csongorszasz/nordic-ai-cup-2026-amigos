"""Grouped calibration of valid aligner spans, never of incumbent fallbacks."""

import argparse
import hashlib
import json
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.modernbert_data import grouped_folds
from benchmark import paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs
from calibrate_spans import GRID, fit_offsets
from check_alignment_tokens import MODEL, REVISION
from llm_probe import score_records
from train_quote_oof import validate_coverage


def apply_alignment_offsets(rows, offsets):
    return [
        {
            **row,
            "span": adjusted_span(row["span"], offsets, row["duration"])
            if row["answer"] and row["alignment_reason"] == "retimed" else row["span"],
        }
        for row in rows
    ]


def cross_validate_alignment(rows, excluded, seed):
    selected = [row for row in rows if row["transcript_id"] not in excluded]
    tids = {row["transcript_id"] for row in selected}
    if len(tids) < 5:
        raise ValueError("Alignment calibration requires at least five non-demo conversations.")
    folds = grouped_folds(tids, 5, seed)
    predictions, reports = [], []
    for held_out in folds:
        held = set(held_out)
        training = [row for row in selected if row["transcript_id"] not in held]
        testing = [row for row in selected if row["transcript_id"] in held]
        eligible = [row for row in training if row["answer"] and row["alignment_reason"] == "retimed"]
        offsets = fit_offsets(eligible, GRID)
        predictions.extend(apply_alignment_offsets(testing, offsets))
        reports.append({
            "training_tids": sorted({row["transcript_id"] for row in training}),
            "fitting_question_ids": [row["question_id"] for row in eligible],
            "held_out_tids": held_out, "offsets": list(offsets),
        })
    validate_coverage(predictions, selected)
    return predictions, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/alignment_calibration"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Calibration output already contains a run; use a fresh directory.")
    baseline, requests, _, _ = load_inputs(args.baseline)
    source = json.loads((args.aligned / "summary.json").read_text())
    if (
        not source["complete"] or source["smoke_only"] or source["questions"] != len(baseline)
        or not source["alignment_success"] or source["conversation_failures"]
        or not source.get("timestamp_classes_bounded_by_audio")
        or source["model"] != MODEL or source["revision"] != REVISION
        or source["aligner_offsets"] != [0.0, 0.0]
    ):
        raise ValueError("A complete successful, bounded, uncalibrated alignment run is required.")
    path = args.aligned / "questions.json"
    rows = json.loads(path.read_text())
    validate_coverage(rows, baseline)
    original = {row["question_id"]: row for row in baseline}
    for row in rows:
        if any(
            row[key] != original[row["question_id"]][key]
            for key in ("question", "question_type", "quote", "word_range", "duration")
        ):
            raise ValueError(f"Alignment changed a frozen input: {row['question_id']}")
        if row["alignment_reason"] not in ("retimed", "invalid_span_keep", "base_no"):
            raise ValueError(f"Unexpected alignment status: {row['alignment_reason']}")
    excluded = {tid for entry in requests for tid in entry["demonstration_tids"]}
    if sorted(excluded) != source["excluded_demonstration_tids"]:
        raise ValueError("Alignment and baseline demonstration provenance differ.")
    retained = [baseline_prediction(row) for row in baseline if row["transcript_id"] not in excluded]
    raw = [row for row in rows if row["transcript_id"] not in excluded]
    report = {
        "diagnostic_only": True, "serving_artifact_written": False,
        "source_model": MODEL, "source_revision": REVISION,
        "source_device": source["device"], "source_dtype": source["dtype"],
        "source_questions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "excluded_demonstration_tids": sorted(excluded),
        "questions": len(raw), "baseline": score_records(retained), "raw_aligned": score_records(raw),
        "raw_incumbent": score_records([
            row for row in baseline if row["transcript_id"] not in excluded
        ]),
        "grid": [list(pair) for pair in GRID],
        "applicability": "Predicted yes with alignment_reason=retimed; keep fallback spans unchanged.",
        "seeds": {}, "promotion": "Not a GPU/HTTP gate or a compatible serving calibration artifact.",
    }
    for seed in (13, 37):
        predictions, folds = cross_validate_alignment(rows, excluded, seed)
        report["seeds"][str(seed)] = {
            "calibrated": score_records(predictions),
            "versus_incumbent": paired_comparison(retained, predictions, seed=seed),
            "versus_raw_aligned": paired_comparison(raw, predictions, seed=seed),
            "folds": folds,
        }
        write_json(args.output / f"oof_seed_{seed}.json", predictions)
    write_json(args.output / "summary.json", report)
    print(json.dumps({**report, "seeds": {
        seed: {key: value for key, value in result.items() if key != "folds"}
        for seed, result in report["seeds"].items()
    }}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
