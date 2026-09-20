"""Grouped, demonstration-disjoint calibration of evidence boundary offsets.

The input is a completed benchmark run. Run this on IDUN; it uses saved
predictions, never evaluation captures or additional model calls.
"""

import argparse
import hashlib
import json
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.modernbert_data import grouped_folds, load_rows
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from utils import gold_evidence, temporal_iou

OFFSETS = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, 0.8)
GRID = [(left, right) for left in OFFSETS for right in OFFSETS]


def apply_offsets(records, offsets):
    return [
        {
            **row,
            "span": adjusted_span(row["span"], offsets, row.get("duration"))
            if row["answer"] else None,
            "original_span": row["span"],
            "offsets": list(offsets),
        }
        for row in records
    ]


def fit_offsets(records, grid=GRID):
    positives = [row for row in records if row["label"] == 1 and row["gold"] is not None]
    if not positives:
        raise ValueError("Calibration requires annotated training positives.")

    def objective(offsets):
        return sum(
            temporal_iou(
                row["gold"],
                adjusted_span(row["span"], offsets, row.get("duration"))
                if row["answer"] else None,
            )
            for row in positives
        )

    return max(
        grid, key=lambda offsets: (objective(offsets), -sum(abs(value) for value in offsets))
    )


def cross_validate(records, excluded_tids, n_folds=5, seed=13, grid=GRID):
    selected = [row for row in records if row["transcript_id"] not in excluded_tids]
    tids = sorted({row["transcript_id"] for row in selected})
    folds = grouped_folds(tids, min(n_folds, len(tids)), seed)
    predictions, fold_report = [], []
    for held_out in folds:
        held = set(held_out)
        train = [row for row in selected if row["transcript_id"] not in held]
        test = [row for row in selected if row["transcript_id"] in held]
        offsets = fit_offsets(train, grid)
        predictions.extend(apply_offsets(test, offsets))
        fold_report.append({
            "train_tids": sorted({row["transcript_id"] for row in train}),
            "held_out_tids": held_out, "offsets": list(offsets),
        })
    return selected, predictions, fold_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="results/span_calibration")
    parser.add_argument("--start-grid", type=float, nargs="+", default=list(OFFSETS))
    parser.add_argument("--end-grid", type=float, nargs="+", default=list(OFFSETS))
    parser.add_argument("--baseline-offsets", type=float, nargs=2, default=[0.0, 0.0])
    args = parser.parse_args()
    record_path, request_path = Path(args.records), Path(args.requests)
    records = json.loads(record_path.read_text())
    requests = json.loads(request_path.read_text())
    expected = {row["question_id"]: row for row in load_rows()}
    if (
        len(records) != len(expected)
        or {row["question_id"] for row in records} != expected.keys()
    ):
        parser.error("Calibration requires every supplied question exactly once.")
    for row in records:
        truth = expected[row["question_id"]]
        gold = gold_evidence(truth)
        if (
            row["transcript_id"] != truth["transcript_id"]
            or row["label"] != int(truth["label"])
            or row["question_type"] != truth["question_type"]
            or row["gold"] != (list(gold) if gold is not None else None)
        ):
            parser.error(f"Prediction provenance disagrees with training data: {row['question_id']}")
    if any("demonstration_tids" not in item for item in requests):
        parser.error("Input lacks demonstration provenance; regenerate with benchmark.py.")
    request_tids = {item["transcript_id"] for item in requests}
    if request_tids != {row["transcript_id"] for row in records}:
        parser.error("Prediction and request manifests cover different conversations.")
    excluded = {tid for item in requests for tid in item["demonstration_tids"]}
    import math

    if any(not math.isfinite(value) for value in args.start_grid + args.end_grid + args.baseline_offsets):
        parser.error("Offset values must be finite.")
    grid = [(start, end) for start in args.start_grid for end in args.end_grid]
    raw_baseline, predictions, folds = cross_validate(records, excluded, args.folds, args.seed, grid)
    baseline = apply_offsets(raw_baseline, args.baseline_offsets)
    summary = {
        "method": "constant-boundary-offsets",
        "excluded_demonstration_tids": sorted(excluded),
        "baseline": score_records(baseline),
        "candidate": score_records(predictions),
        "paired": paired_comparison(baseline, predictions, seed=args.seed),
        "folds": folds, "seed": args.seed,
        "grid": [list(offsets) for offsets in grid],
        "baseline_offsets": args.baseline_offsets,
        "input_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "candidate_offsets_full_training": list(fit_offsets(records, grid)),
        "promotion": "requires confirmation and uncached endpoint replay",
    }
    output = Path(args.output)
    write_json(output / "summary.json", summary)
    write_json(output / "oof_questions.json", predictions)
    source = json.loads(
        record_path.with_name(record_path.name.replace("_questions.json", "_summary.json")).read_text()
    )
    configs = {request["asr_config_hash"] for request in requests}
    if len(configs) != 1 or not source.get("complete") or source["questions"] != len(records):
        parser.error("Calibration requires a complete, single-ASR benchmark.")
    offsets = summary["candidate_offsets_full_training"]
    write_json(output / "calibration.json", {
        "method": "constant-boundary-offsets",
        "start_offset_s": offsets[0], "end_offset_s": offsets[1],
        "asr_config_hash": next(iter(configs)),
        "model": source["model"], "revision": source["revision"],
        "variant": source["variant"],
        "source_records_sha256": summary["input_sha256"],
    })
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
