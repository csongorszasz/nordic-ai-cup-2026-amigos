"""Complete fixed quote-localizer OOF folds, reusing the unchanged pilot fold."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.lora_data import split_rows
from benchmark import paired_comparison, write_json
from llm_probe import score_records

ROOT = Path(__file__).resolve().parent
FROZEN_SOURCES = (
    "train_quote_lora.py", "answerers/lora_data.py", "answerers/llm_client.py",
    "answerers/llm_prompt.py", "answerers/llm_parse.py", "answerers/align.py",
    "answerers/boundaries.py", "answerers/base.py", "answerers/modernbert_data.py",
    "benchmark.py", "llm_probe.py", "local_evaluator.py", "utils.py",
)


def validate_coverage(predictions, expected):
    ids = [row["question_id"] for row in predictions]
    if len(ids) != len(set(ids)) or set(ids) != {row["question_id"] for row in expected}:
        raise ValueError("OOF predictions have missing or repeated questions.")
    reference = {row["question_id"]: row for row in expected}
    for row in predictions:
        source = reference[row["question_id"]]
        if any(row[key] != source[key] for key in ("transcript_id", "label", "prediction", "answer", "gold")):
            raise ValueError(f"OOF output changed a frozen decision/reference: {row['question_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/lora_oof"))
    args = parser.parse_args()
    manifest = json.loads((args.pilot_run / "source_manifest.json").read_text())
    for name in FROZEN_SOURCES:
        digest = hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if manifest["files"].get(name) != digest:
            raise ValueError(f"Pilot training/inference code changed: {name}")
    request = json.loads((args.pilot_run / "run_request.json").read_text())
    pilot_parser = argparse.ArgumentParser(add_help=False)
    pilot_parser.add_argument("--fold", type=int, default=0)
    pilot_parser.add_argument("--seed", type=int, default=13)
    pilot_parser.add_argument("--epochs", type=int, default=2)
    pilot_parser.add_argument("--learning-rate", type=float, default=1e-4)
    pilot_parser.add_argument("--accumulation", type=int, default=4)
    pilot_args, _ = pilot_parser.parse_known_args(request["arguments"])
    if vars(pilot_args) != {"fold": 0, "seed": 13, "epochs": 2, "learning_rate": 1e-4, "accumulation": 4}:
        raise ValueError("Pilot hyperparameters do not match the preregistered OOF setup.")
    rows = json.loads((args.baseline / "base_legacy_questions.json").read_text())
    requests = json.loads((args.baseline / "base_legacy_conversations.json").read_text())
    excluded = {tid for entry in requests for tid in entry["demonstration_tids"]}
    expected = [row for row in rows if row["transcript_id"] not in excluded]
    _, _, folds = split_rows(rows, excluded, 0, 13)
    args.output.mkdir(parents=True, exist_ok=True)
    adapted, unadapted, fold_summaries = [], [], []
    for index, held_out in enumerate(folds):
        if index == 0:
            directory = args.pilot_run / "results" / "lora_pilot"
        else:
            directory = args.output / f"fold_{index}"
            with (args.output / f"fold_{index}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable, str(ROOT / "train_quote_lora.py"),
                        "--baseline", str(args.baseline), "--fold", str(index),
                        "--seed", "13", "--epochs", "2", "--learning-rate", "0.0001",
                        "--accumulation", "4", "--output", str(directory),
                    ],
                    check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
        split = json.loads((directory / "split.json").read_text())
        if (
            set(split["validation_tids"]) != set(held_out)
            or set(split["training_tids"]) & (set(held_out) | excluded)
            or split["excluded_demonstration_tids"] != sorted(excluded)
        ):
            raise ValueError(f"Fold {index} provenance is inconsistent.")
        after = json.loads((directory / "adapted_questions.json").read_text())
        before = json.loads((directory / "unadapted_questions.json").read_text())
        held_rows = [row for row in expected if row["transcript_id"] in held_out]
        validate_coverage(after, held_rows)
        validate_coverage(before, held_rows)
        adapted.extend(after)
        unadapted.extend(before)
        summary = json.loads((directory / "summary.json").read_text())
        fold_summaries.append({
            "fold": index, "baseline": summary["baseline"], "adapted": summary["adapted"],
            "unadapted": summary["unadapted"], "reasons": summary["adapted_reasons"],
            "max_estimated_combined_s": summary["max_estimated_combined_s"],
        })
        write_json(args.output / "adapted_oof.json", adapted)
        write_json(args.output / "unadapted_oof.json", unadapted)
        print(json.dumps(fold_summaries[-1]), flush=True)
    validate_coverage(adapted, expected)
    validate_coverage(unadapted, expected)
    baseline = [
        {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row.get("duration")) if row["answer"] else None}
        for row in expected
    ]
    summary = {
        "complete_oof": True, "questions": len(expected), "folds": fold_summaries,
        "excluded_demonstration_tids": sorted(excluded),
        "baseline": score_records(baseline), "unadapted": score_records(unadapted),
        "adapted": score_records(adapted),
        "versus_baseline": paired_comparison(baseline, adapted),
        "versus_unadapted": paired_comparison(unadapted, adapted),
        "max_estimated_combined_s": max(item["max_estimated_combined_s"] for item in fold_summaries),
        "latency_note": "Separate-run sum, not a serving acceptance measurement.",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "folds"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
