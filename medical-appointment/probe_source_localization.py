"""Bounded source-unit geometry probe; reference-assisted ceilings are not scores."""

import argparse
import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict
from pathlib import Path

from answerers.evidence_units import (
    SOURCE_RECIPE, build_evidence_units, incumbent_unit, shortlist_evidence,
)
from audit_localization import exact_score
from benchmark import write_json
from benchmark_alignment import baseline_prediction, load_inputs
from train_quote_oof import validate_coverage
from utils import temporal_iou


def validate_audit_replay(rows, audited, summary, recipe):
    predictions = [baseline_prediction(row) for row in rows]
    validate_coverage(audited, predictions)
    recorded = {row["question_id"]: row for row in audited}
    if any(row["span"] != recorded[row["question_id"]]["span"] for row in predictions):
        raise ValueError("The annotation audit used different incumbent spans.")
    actual = exact_score(predictions)
    if any(not math.isclose(actual[key], summary["incumbent"][key], rel_tol=0, abs_tol=1e-12)
           for key in ("score", "mean_tiou", "accuracy", "questions", "positives")):
        raise ValueError("The annotation audit's exact incumbent does not reproduce.")
    if (
        summary.get("diagnostic_only") is not True
        or recipe.get("version") != 1
        or recipe.get("planned_source_shortlist") != SOURCE_RECIPE["shortlist_units"]
        or recipe.get("offsets_s") != SOURCE_RECIPE["offsets_s"]
    ):
        raise ValueError("Source candidate budgets or timing differ from the frozen audit recipe.")
    return predictions


def build_runtime_case(row, transcript, inventory):
    baseline = baseline_prediction(row)
    incumbent = (
        incumbent_unit(transcript["words"], row["duration"], baseline["span"], row["word_range"])
        if row["answer"] else None
    )
    selected = shortlist_evidence(
        row["question"], inventory, row["duration"], incumbent=incumbent,
        top_k=SOURCE_RECIPE["shortlist_units"],
    )
    return {
        "question_id": row["question_id"], "transcript_id": row["transcript_id"],
        "question": row["question"], "baseline_answer": row["answer"],
        "baseline_span": baseline["span"],
        "candidates": [
            {**asdict(unit), "resolved_span": unit.resolved_span(row["duration"])} for unit in selected
        ],
    }, incumbent, selected


def best_unit(units, gold, duration):
    if not units:
        return {"tiou": 0.0, "span": None, "word_range": None, "families": [], "unit_index": None}
    ranked = [(unit, unit.resolved_span(duration)) for unit in units]
    unit, span = max(ranked, key=lambda item: (
        temporal_iou(gold, item[1]), -(item[1][1] - item[1][0]), -item[1][0],
    ))
    return {
        "tiou": temporal_iou(gold, span), "span": span,
        "word_range": list(unit.word_range()), "families": list(unit.families), "unit_index": unit.index,
    }


def probe_geometry(rows, transcripts):
    inventories = {
        tid: build_evidence_units(transcript["words"], transcript["duration"])
        for tid, transcript in transcripts.items()
    }
    runtime, diagnostics = [], []
    for row in rows:
        tid, duration = row["transcript_id"], row["duration"]
        entry, incumbent, selected = build_runtime_case(row, transcripts[tid], inventories[tid])
        runtime.append(entry)
        pool = [*inventories[tid], *([incumbent] if incumbent is not None else [])]
        diagnostic = {
            "question_id": row["question_id"], "transcript_id": tid,
            "label": row["label"], "gold": row["gold"], "baseline_answer": row["answer"],
            "pool_units": len(pool), "shortlist_units": len(selected), "oracles": None,
        }
        if row["label"] == 1:
            diagnostic["oracles"] = {
                "all_units": best_unit(pool, row["gold"], duration),
                "shortlisted_units": best_unit(selected, row["gold"], duration),
                **{
                    family: best_unit([unit for unit in pool if family in unit.families], row["gold"], duration)
                    for family in ("clause", "sentence", "episode", "incumbent")
                },
            }
        diagnostics.append(diagnostic)
    return runtime, diagnostics, inventories


def geometry_summary(diagnostics):
    positive = [row for row in diagnostics if row["label"] == 1]
    if not positive:
        raise ValueError("Candidate coverage requires reference positives.")
    return {
        "questions": len(diagnostics), "positives": len(positive),
        "shortlist_mean": statistics.fmean(row["shortlist_units"] for row in diagnostics),
        "shortlist_max": max(row["shortlist_units"] for row in diagnostics),
        "candidate_oracles": {
            name: {
                "all_positive_mean_tiou": statistics.fmean(row["oracles"][name]["tiou"] for row in positive),
                "frozen_decision_mean_tiou": statistics.fmean(
                    row["oracles"][name]["tiou"] if row["baseline_answer"] else 0.0 for row in positive
                ),
                "at_least_0_8": sum(row["oracles"][name]["tiou"] >= 0.8 for row in positive),
                "at_least_0_95": sum(row["oracles"][name]["tiou"] >= 0.95 for row in positive),
                "positive_denominator": len(positive),
            }
            for name in positive[0]["oracles"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/source_localization"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh source-localization output directory.")
    rows, requests, transcripts, _ = load_inputs(args.baseline)
    audited = json.loads((args.audit / "questions.json").read_text())
    summary = json.loads((args.audit / "summary.json").read_text())
    recipe = json.loads((args.audit / "frozen_recipe.json").read_text())
    predictions = validate_audit_replay(rows, audited, summary, recipe)
    for name in ("base_legacy_questions.json", "base_legacy_conversations.json"):
        digests = [digest for path, digest in recipe["source_sha256"].items() if Path(path).name == name]
        actual = hashlib.sha256((args.baseline / name).read_bytes()).hexdigest()
        if digests != [actual]:
            raise ValueError(f"Frozen annotation-audit inputs changed: {name}")
    excluded = sorted({tid for request in requests for tid in request["demonstration_tids"]})
    if excluded != recipe["excluded_demonstration_tids"]:
        raise ValueError("Demonstration-source exclusions changed after the annotation audit.")
    write_json(args.output / "frozen_recipe.json", {
        **SOURCE_RECIPE, "job_id": os.environ.get("SLURM_JOB_ID"),
        "audit_recipe_sha256": hashlib.sha256((args.audit / "frozen_recipe.json").read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256((args.baseline / "base_legacy_questions.json").read_bytes()).hexdigest(),
        "outer_folds": recipe["outer_folds"], "excluded_demonstration_tids": excluded,
        "mode": "geometry_only", "serving_artifact_written": False,
    })
    runtime, diagnostics, inventories = probe_geometry(rows, transcripts)
    report = {
        "diagnostic_only": True, "mode": "geometry_only", "serving_artifact_written": False,
        "incumbent": exact_score(predictions),
        "all_questions": geometry_summary(diagnostics),
        "demo_disjoint": geometry_summary([row for row in diagnostics if row["transcript_id"] not in excluded]),
        "excluded_demonstration_tids": excluded,
        "inventory_counts": {tid: len(units) for tid, units in inventories.items()},
        "limitations": [
            "Reference-assisted unit coverage is not a model score or an achieved improvement.",
            "The lexical shortlist seeds a source-ranker experiment; it never changes a boolean.",
            "Episode candidates are contiguous sentence sequences, not verified clinical actions.",
            "Interpretation context is not added to a citation when measuring its IoU.",
            "All reference positives remain in frozen-decision denominators, including missed positives.",
        ],
    }
    write_json(args.output / "runtime_candidates.json", runtime)
    write_json(args.output / "diagnostics.json", diagnostics)
    write_json(args.output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "inventory_counts"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
