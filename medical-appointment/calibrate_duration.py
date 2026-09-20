"""Low-capacity duration-aware boundary calibration on grouped saved outputs."""

import argparse
import json
import math
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.modernbert_data import grouped_folds
from benchmark import paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs
from llm_probe import score_records
from train_quote_oof import validate_coverage
from utils import temporal_iou


BASELINE = (0.2, 0.0, 0.0, 0.0)
GRID = [
    (start, end, left, right)
    for start in (0.2, 0.0) for end in (0.0, -0.2)
    for left in (0.0, 0.025, 0.05, 0.1) for right in (0.0, 0.025, 0.05, 0.1)
]


def transform_span(span, parameters, duration):
    if len(parameters) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("Four finite boundary parameters are required.")
    if span is None:
        return None
    length = span[1] - span[0]
    start, end, left_fraction, right_fraction = parameters
    return adjusted_span(
        span, (start + left_fraction * length, end - right_fraction * length), duration,
    )


def apply_parameters(rows, parameters):
    return [
        {**row, "span": transform_span(row["span"], parameters, row["duration"]) if row["answer"] else None}
        for row in rows
    ]


def fit_parameters(rows):
    positives = [row for row in rows if row["label"] == 1 and row["gold"] is not None]
    if not positives:
        raise ValueError("Calibration requires training positives.")

    def objective(parameters):
        value = sum(
            temporal_iou(
                row["gold"], transform_span(row["span"], parameters, row["duration"]) if row["answer"] else None,
            )
            for row in positives
        )
        deviation = sum(abs(value - base) for value, base in zip(parameters, BASELINE))
        return value, -deviation

    return max(GRID, key=objective)


def cross_validate(rows, excluded, seed):
    selected = [row for row in rows if row["transcript_id"] not in excluded]
    tids = {row["transcript_id"] for row in selected}
    if len(tids) < 5:
        raise ValueError("Use at least five non-demo conversations.")
    predictions, reports = [], []
    for held_out in grouped_folds(tids, 5, seed):
        held = set(held_out)
        training = [row for row in selected if row["transcript_id"] not in held]
        validation = [row for row in selected if row["transcript_id"] in held]
        parameters = fit_parameters(training)
        predictions.extend(apply_parameters(validation, parameters))
        reports.append({
            "training_tids": sorted({row["transcript_id"] for row in training}),
            "held_out_tids": held_out, "parameters": list(parameters),
        })
    validate_coverage(predictions, selected)
    return predictions, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/duration_calibration"))
    args = parser.parse_args()
    rows, requests, _, _ = load_inputs(args.baseline)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh diagnostic output directory.")
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    baseline = [baseline_prediction(row) for row in rows if row["transcript_id"] not in excluded]
    report = {
        "diagnostic_only": True, "serving_artifact_written": False,
        "parameter_order": ["start_offset_s", "end_offset_s", "start_trim_fraction", "end_trim_fraction"],
        "grid": [list(parameters) for parameters in GRID],
        "baseline_parameters": list(BASELINE),
        "excluded_demonstration_tids": sorted(excluded), "baseline": score_records(baseline), "seeds": {},
    }
    for seed in (13, 37):
        predictions, folds = cross_validate(rows, excluded, seed)
        report["seeds"][str(seed)] = {
            "candidate": score_records(predictions),
            "paired": paired_comparison(baseline, predictions, seed=seed),
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
