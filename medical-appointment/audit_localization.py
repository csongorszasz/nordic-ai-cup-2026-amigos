"""Reproducible annotation/representation audit; never a serving policy."""

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from answerers.align import align_quote_matches
from answerers.modernbert_data import grouped_folds
from answerers.passages import build_sentences, overlap_word_range
from answerers.span_targets import optimal_quote_target, word_text
from benchmark import write_json
from benchmark_alignment import baseline_prediction, load_inputs
from local_evaluator import Statistics
from utils import temporal_iou
from windows import build_windows


RECIPE = {
    "version": 1,
    "offsets_s": [0.2, 0.0],
    "local_oracle_context_words": 24,
    "boundary_grids_s": [0.02, 0.04, 0.08],
    "pause_minimum_s": 0.12,
    "grouped_seeds": [13, 37],
    "fold_count": 5,
    "planned_source_shortlist": 32,
    "planned_acoustic_positions_per_endpoint": 8,
    "labels_unchanged": True,
    "diagnostic_oracles_are_predictions": False,
}


def exact_score(rows):
    score = Statistics()
    for row in rows:
        score.record(row["question_type"], row["label"], row["prediction"], row["gold"], row["span"])
    return {
        "questions": score.total, "positives": len(score.tious), "correct": score.correct,
        "accuracy": score.accuracy, "mean_tiou": score.mean_tiou, "score": score.final_score,
    }


def validate_qualified(predictions, qualified, summary):
    expected = {row["question_id"]: row for row in predictions}
    actual = {row["question_id"]: row for row in qualified}
    if len(expected) != len(predictions) or len(actual) != len(qualified) or expected.keys() != actual.keys():
        raise ValueError("Qualified replay must contain the same complete unique questions.")
    for qid, row in expected.items():
        other = actual[qid]
        if (
            not isinstance(other["answer"], bool)
            or any(row[key] != other[key] for key in (
                "transcript_id", "label", "question_type", "answer", "prediction", "gold",
            ))
            or other.get("request_error") is not None
        ):
            raise ValueError(f"Qualified decision/reference mismatch: {qid}")
        left, right = row["span"], other["span"]
        if left is None or right is None:
            equal = left is None and right is None
        else:
            equal = (
                isinstance(right, list) and len(right) == 2
                and all(
                    not isinstance(b, bool) and isinstance(b, (int, float))
                    and math.isfinite(b) and abs(a - b) <= 1e-6
                    for a, b in zip(left, right)
                )
            )
        if not equal:
            raise ValueError(f"Qualified span mismatch or repeated calibration: {qid}")
    score = exact_score(predictions)
    if (
        summary.get("complete") is not True or summary.get("full_corpus") is not True
        or summary.get("aborted") is not False or summary.get("failed_conversations") != 0
        or summary.get("timeouts") != 0 or summary.get("questions") != len(predictions)
    ):
        raise ValueError("The qualified replay did not complete successfully.")
    for name in ("score", "accuracy", "mean_tiou"):
        if not math.isclose(score[name], summary[name], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Qualified exact {name} does not reproduce.")
    return score


def validate_word_clock(words, duration):
    if not words or isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("The audit requires words and a finite positive duration.")
    previous_start = 0.0
    for word in words:
        start, end = word["start"], word["end"]
        if (
            not isinstance(word["word"], str) or not word["word"].strip()
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in (start, end))
            or not previous_start <= start <= end <= duration
        ):
            raise ValueError("The audit requires finite ordered word times within the audio.")
        previous_start = start


def _distribution(values):
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values), "minimum": min(values), "median": statistics.median(values),
        "mean": statistics.fmean(values), "maximum": max(values),
    }


def _word_oracle(words, gold, duration, bounds=None, offsets=(0.0, 0.0)):
    first, last = bounds if bounds is not None else (0, len(words) - 1)
    target = optimal_quote_target(words[first:last + 1], gold, duration, unique=False, offsets=offsets)
    if target is None:
        return {"tiou": 0.0, "span": None, "word_range": None, "quote": None}
    return {
        "tiou": target["tiou"], "span": target["span"], "quote": target["quote"],
        "word_range": [target["first_word"] + first, target["last_word"] + first],
    }


def _unit_oracle(units, gold):
    if not units:
        return {"tiou": 0.0, "span": None, "word_range": None}
    unit = max(units, key=lambda item: (
        temporal_iou(gold, item.span()), -(item.last_word - item.first_word), -item.first_word,
    ))
    return {
        "tiou": temporal_iou(gold, unit.span()), "span": list(unit.span()),
        "word_range": [unit.first_word, unit.last_word],
    }


def audit_question(row, transcript, units):
    result = dict(baseline_prediction(row))
    if row["label"] != 1:
        result["audit"] = None
        return result
    gold, words, span = row["gold"], transcript["words"], result["span"]
    overlap = overlap_word_range(words, *gold)
    length = gold[1] - gold[0]
    gap = max(0.0, gold[0] - span[1], span[0] - gold[1]) if span is not None else None
    start_distance = min(abs(word["start"] - gold[0]) for word in words)
    end_distance = min(abs(word["end"] - gold[1]) for word in words)
    pauses = [
        value for left, right in zip(words, words[1:])
        if right["start"] - left["end"] >= RECIPE["pause_minimum_s"]
        for value in (left["end"], right["start"])
    ]
    oracles = {
        "raw_global_words": _word_oracle(words, gold, row["duration"]),
        "corrected_global_words": _word_oracle(words, gold, row["duration"], offsets=(0.2, 0.0)),
        **{name: _unit_oracle(candidates, gold) for name, candidates in units.items()},
    }
    for name, margin in (("corrected_inside_quote", 0), ("corrected_quote_plus_24", 24)):
        anchor = row.get("word_range") if row["answer"] else None
        oracles[name] = (
            _word_oracle(
                words, gold, row["duration"],
                (max(0, anchor[0] - margin), min(len(words) - 1, anchor[1] + margin)),
                offsets=(0.2, 0.0),
            ) if anchor is not None else
            {"tiou": 0.0, "span": None, "word_range": None, "quote": None}
        )
    matches = align_quote_matches(words, row.get("quote") or "") if row["answer"] else []
    result["audit"] = {
        "gold_duration_s": length, "incumbent_tiou": temporal_iou(gold, span),
        "missed_positive": not row["answer"],
        "disjoint_answered_yes": span is not None and temporal_iou(gold, span) == 0,
        "disjoint_gap_s": gap,
        "incumbent_to_gold_duration_ratio": (span[1] - span[0]) / length if span is not None else None,
        "nearest_start_word_boundary_s": start_distance,
        "nearest_end_word_boundary_s": end_distance,
        "nearest_start_pause_boundary_s": min(abs(value - gold[0]) for value in pauses) if pauses else None,
        "nearest_end_pause_boundary_s": min(abs(value - gold[1]) for value in pauses) if pauses else None,
        "overlapping_word_range": list(overlap) if overlap is not None else None,
        "overlapping_word_text": word_text(words, *overlap) if overlap is not None else None,
        "identical_quote_occurrences": [
            {"span": list(match[:2]), "word_range": list(match[2:])} for match in matches
        ],
        "oracles": oracles,
    }
    return result


def audit_records(rows, requests, transcripts):
    units = {}
    for tid, transcript in transcripts.items():
        validate_word_clock(transcript["words"], transcript["duration"])
        units[tid] = {
            "legacy_windows": build_windows(transcript["words"]),
            "merged_sentences": build_sentences(transcript["words"]),
        }
    audited = [audit_question(row, transcripts[row["transcript_id"]], units[row["transcript_id"]]) for row in rows]
    positives = [row for row in audited if row["label"] == 1]
    if not positives:
        raise ValueError("The annotation audit requires reference positives.")
    excluded = sorted({tid for request in requests for tid in request["demonstration_tids"]})
    endpoints = [value for row in positives for value in row["gold"]]
    entries = [row["audit"] for row in positives]
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[(row["transcript_id"], row["question"])].append(row)
    duplicate_groups = [
        {
            "transcript_id": tid, "question": question,
            "question_ids": [row["question_id"] for row in group],
            "conflicting_reference": len({(row["label"], tuple(row["gold"] or ())) for row in group}) > 1,
        }
        for (tid, question), group in duplicates.items() if len(group) > 1
    ]
    report = {
        "diagnostic_only": True, "serving_artifact_written": False,
        "incumbent": exact_score(audited),
        "demo_disjoint_incumbent": exact_score([row for row in audited if row["transcript_id"] not in excluded]),
        "excluded_demonstration_tids": excluded,
        "reference": {
            "positive_questions": len(positives), "endpoints": len(endpoints),
            "endpoint_grid_counts": {
                str(grid): sum(abs(value / grid - round(value / grid)) <= 1e-7 for value in endpoints)
                for grid in RECIPE["boundary_grids_s"]
            },
            "duration_s": _distribution([entry["gold_duration_s"] for entry in entries]),
            "nearest_start_word_boundary_s": _distribution([entry["nearest_start_word_boundary_s"] for entry in entries]),
            "nearest_end_word_boundary_s": _distribution([entry["nearest_end_word_boundary_s"] for entry in entries]),
            "exact_word_start_count": sum(entry["nearest_start_word_boundary_s"] <= 1e-8 for entry in entries),
            "exact_word_end_count": sum(entry["nearest_end_word_boundary_s"] <= 1e-8 for entry in entries),
            "duplicate_question_groups": duplicate_groups,
        },
        "overlapping_error_counts": {
            "missed_positives": sum(entry["missed_positive"] for entry in entries),
            "disjoint_answered_yes": sum(entry["disjoint_answered_yes"] for entry in entries),
            "disjoint_gap_at_least_2s": sum(entry["disjoint_gap_s"] is not None and entry["disjoint_gap_s"] >= 2 for entry in entries),
            "span_at_least_twice_gold": sum(entry["incumbent_to_gold_duration_ratio"] is not None and entry["incumbent_to_gold_duration_ratio"] >= 2 for entry in entries),
            "span_at_most_half_gold": sum(entry["incumbent_to_gold_duration_ratio"] is not None and entry["incumbent_to_gold_duration_ratio"] <= 0.5 for entry in entries),
        },
        "oracles": {
            name: {
                "all_positive_mean_tiou": statistics.fmean(entry["oracles"][name]["tiou"] for entry in entries),
                "frozen_decision_mean_tiou": statistics.fmean(
                    0.0 if entry["missed_positive"] else entry["oracles"][name]["tiou"] for entry in entries
                ),
                "positive_denominator": len(entries),
                "requires_incumbent_anchor": name in {"corrected_inside_quote", "corrected_quote_plus_24"},
            }
            for name in entries[0]["oracles"]
        },
        "limitations": [
            "Exact quote repetitions are not a semantic enumeration of all supporting occurrences.",
            "All oracle fields use reference spans and are diagnostic, never inference-time features.",
            "The annotation grid does not establish the generator or acoustic boundary accuracy.",
            "Observed source/extent/timing losses overlap; oracle gains are not additive.",
            "Grouped partitions of this repeatedly inspected corpus are not virgin holdouts.",
        ],
    }
    return audited, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--qualified", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/localization_audit"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh annotation-audit output directory.")
    rows, requests, transcripts, _ = load_inputs(args.baseline)
    qualified = json.loads((args.qualified / "questions.json").read_text())
    summary = json.loads((args.qualified / "summary.json").read_text())
    validate_qualified([baseline_prediction(row) for row in rows], qualified, summary)
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    tids = {row["transcript_id"] for row in rows if row["transcript_id"] not in excluded}
    recipe = {
        **RECIPE, "excluded_demonstration_tids": sorted(excluded),
        "outer_folds": {str(seed): grouped_folds(tids, 5, seed) for seed in RECIPE["grouped_seeds"]},
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.baseline / "base_legacy_questions.json",
                args.baseline / "base_legacy_conversations.json",
                args.qualified / "questions.json", args.qualified / "summary.json",
                Path("data/question_train.csv"),
            )
        },
        "request_provenance": requests, "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(args.output / "frozen_recipe.json", recipe)
    audited, report = audit_records(rows, requests, transcripts)
    write_json(args.output / "questions.json", audited)
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
