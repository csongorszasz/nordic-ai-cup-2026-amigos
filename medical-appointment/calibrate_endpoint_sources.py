"""Cross-fit four endpoint-source policies over the same frozen word occurrence."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from answerers.acoustic_boundaries import MODEL, RECIPE, REVISION
from answerers.base import normalize_answer
from answerers.modernbert_data import grouped_folds
from audit_localization import exact_score
from benchmark import paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs
from train_quote_oof import validate_coverage
from utils import temporal_iou


POLICIES = {
    "incumbent": (False, False),
    "ctc_end": (False, True),
    "ctc_start": (True, False),
    "ctc_both": (True, True),
}


def apply_policy(baseline, aligned, policy):
    if policy not in POLICIES:
        raise ValueError("Unknown endpoint-source policy.")
    if baseline["word_range"] != aligned["word_range"] or baseline["answer"] != aligned["answer"]:
        raise ValueError("Endpoint mixing must preserve the same decision and source occurrence.")
    output = {**baseline, "endpoint_policy": policy, "endpoint_reason": "incumbent_keep"}
    if policy == "incumbent" or not baseline["answer"] or aligned["ctc_reason"] != "aligned":
        return output
    if temporal_iou(baseline["span"], aligned["span"]) <= 0:
        return {**output, "endpoint_reason": "disjoint_clocks_keep"}
    start_new, end_new = POLICIES[policy]
    span = [
        (aligned if start_new else baseline)["span"][0],
        (aligned if end_new else baseline)["span"][1],
    ]
    if not 0 <= span[0] < span[1] <= baseline["duration"]:
        raise ValueError("Overlapping valid source clocks produced an invalid interval.")
    return {**output, "span": span, "endpoint_reason": "mixed"}


def fit_policy(pairs):
    positives = [(base, aligned) for base, aligned in pairs if base["label"] == 1]
    if not positives:
        raise ValueError("Endpoint-source fitting requires training positives.")
    return max(POLICIES, key=lambda policy: sum(
        temporal_iou(base["gold"], apply_policy(base, aligned, policy)["span"])
        for base, aligned in positives
    ))


def cross_validate(pairs, excluded, seed):
    selected = [(base, aligned) for base, aligned in pairs if base["transcript_id"] not in excluded]
    tids = {base["transcript_id"] for base, _ in selected}
    if len(tids) < 5:
        raise ValueError("Use at least five non-demo conversations.")
    predictions, reports = [], []
    for held_out in grouped_folds(tids, 5, seed):
        held = set(held_out)
        training = [(base, aligned) for base, aligned in selected if base["transcript_id"] not in held]
        testing = [(base, aligned) for base, aligned in selected if base["transcript_id"] in held]
        policy = fit_policy(training)
        predictions.extend(apply_policy(base, aligned, policy) for base, aligned in testing)
        reports.append({
            "training_tids": sorted({base["transcript_id"] for base, _ in training}),
            "held_out_tids": held_out, "policy": policy,
        })
    validate_coverage(predictions, [base for base, _ in selected])
    return predictions, reports


def matched_pairs(raw, rows):
    validate_coverage(rows, raw)
    original = {row["question_id"]: row for row in raw}
    pairs = []
    for row in rows:
        source = original[row["question_id"]]
        baseline = baseline_prediction(source)
        if any(row[key] != source[key] for key in ("question", "question_type", "quote", "word_range", "duration")):
            raise ValueError("The acoustic run changed a frozen source input.")
        reason = row.get("ctc_reason")
        if reason not in {"aligned", "ineligible_keep", "base_no"} or (reason == "base_no") != (not row["answer"]):
            raise ValueError("Endpoint calibration requires explicit successful or ineligible alignment states.")
        valid, checked = normalize_answer((row["answer"], row["span"]), duration=row["duration"])
        if valid != row["answer"] or (list(checked) if checked is not None else None) != row["span"]:
            raise ValueError("An acoustic source row violates the evidence contract.")
        if reason != "aligned" and row["span"] != baseline["span"]:
            raise ValueError("An acoustic fallback changed the qualified incumbent.")
        pairs.append((baseline, row))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/endpoint_sources"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh endpoint-source output directory.")
    raw, requests, _, _ = load_inputs(args.baseline)
    summary = json.loads((args.aligned / "summary.json").read_text())
    provenance = json.loads((args.aligned / "provenance.json").read_text())
    if (
        summary.get("complete") is not True or summary.get("full_corpus") is not True
        or summary.get("smoke_only") is not False or summary.get("questions") != len(raw)
        or summary.get("alignment_feasibility_passed") is not True or summary.get("operational_failures") != []
        or summary.get("word_text_and_anchors_frozen") is not True
        or summary.get("model") != MODEL or summary.get("revision") != REVISION
        or summary.get("recipe") != RECIPE or provenance.get("recipe") != RECIPE
    ):
        raise ValueError("A complete, successful, matching fixed-recipe CTC comparison is required.")
    for name in ("base_legacy_questions.json", "base_legacy_conversations.json"):
        digests = [digest for path, digest in provenance["source_sha256"].items() if Path(path).name == name]
        if digests != [hashlib.sha256((args.baseline / name).read_bytes()).hexdigest()]:
            raise ValueError("The acoustic comparison used different frozen baseline inputs.")
    path = args.aligned / "questions.json"
    rows = json.loads(path.read_text())
    pairs = matched_pairs(raw, rows)
    for name, records in (("baseline", [base for base, _ in pairs]), ("candidate", rows)):
        measured = exact_score(records)
        if any(not math.isclose(measured[key], summary["all_questions"][name][key], abs_tol=1e-12, rel_tol=0)
               for key in ("score", "accuracy", "mean_tiou", "questions", "positives")):
            raise ValueError("The saved acoustic comparison does not reproduce exactly.")
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    if sorted(excluded) != summary["excluded_demonstration_tids"]:
        raise ValueError("The acoustic comparison changed demonstration-source exclusions.")
    selected = [(base, aligned) for base, aligned in pairs if base["transcript_id"] not in excluded]
    retained = [base for base, _ in selected]
    recipe = {
        "version": 1, "policies": {name: list(value) for name, value in POLICIES.items()},
        "policy_tuple": ["use_ctc_start", "use_ctc_end"], "require_clock_overlap": True,
        "seeds": [13, 37], "folds": 5, "incumbent_first_tie_break": True,
        "source_questions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_summary_sha256": hashlib.sha256((args.aligned / "summary.json").read_bytes()).hexdigest(),
        "source_recipe": RECIPE, "source_runtime": summary["runtime"],
        "excluded_demonstration_tids": sorted(excluded),
    }
    write_json(args.output / "frozen_recipe.json", recipe)
    report = {
        "diagnostic_only": True, "serving_artifact_written": False, "deployment_qualified": False,
        "baseline": exact_score(retained), "raw_ctc": exact_score([aligned for _, aligned in selected]),
        "seeds": {}, "excluded_demonstration_tids": sorted(excluded),
        "fixed_policy_diagnostics_not_selection": {
            policy: exact_score([apply_policy(base, aligned, policy) for base, aligned in selected])
            for policy in POLICIES
        },
        "limitations": [
            "Only outer-training conversations select the global endpoint policy; no gold-dependent applicability.",
            "No new offsets or question-specific thresholds are fitted.",
            "No midpoint or enclosing hull is used when source clocks disagree completely.",
            "All ineligible and missed-positive cases retain the incumbent and remain scored.",
            "Repeated development on this corpus is not an independent holdout or a serving acceptance gate.",
        ],
    }
    for seed in recipe["seeds"]:
        predictions, folds = cross_validate(pairs, excluded, seed)
        report["seeds"][str(seed)] = {
            "candidate": exact_score(predictions),
            "paired": paired_comparison(retained, predictions, seed=seed),
            "folds": folds, "reasons": dict(Counter(row["endpoint_reason"] for row in predictions)),
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
