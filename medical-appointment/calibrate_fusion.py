"""Grouped calibration of simple two-citation policies; base decisions stay fixed."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.modernbert_data import grouped_folds
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from utils import temporal_iou


@dataclass(frozen=True)
class Policy:
    mode: str
    overlap: float = 0.0
    weight: float = 0.0


POLICIES = [Policy("keep")]
for overlap in (0.25, 0.5, 0.75):
    POLICIES.extend(Policy("blend", overlap, weight) for weight in (0.25, 0.5, 0.75))
    POLICIES.extend(Policy(mode, overlap) for mode in ("shorter", "longer"))
POLICIES.extend(Policy(mode) for mode in ("earlier", "later"))


def corrected(record):
    return adjusted_span(record["span"], (0.2, 0.0), record.get("duration")) if record["answer"] else None


def fuse(base, alternative, policy):
    first, second = corrected(base), corrected(alternative)
    if first is None or second is None or policy.mode == "keep":
        return first
    overlap = temporal_iou(tuple(first), tuple(second))
    if policy.mode in ("earlier", "later"):
        if overlap != 0.0:
            return first
        if policy.mode == "earlier":
            return min((first, second), key=lambda span: span[0])
        return max((first, second), key=lambda span: span[0])
    if overlap < policy.overlap:
        return first
    if policy.mode == "blend":
        return [
            (1.0 - policy.weight) * a + policy.weight * b
            for a, b in zip(first, second)
        ]
    if policy.mode == "shorter":
        return min((first, second), key=lambda span: span[1] - span[0])
    if policy.mode == "longer":
        return max((first, second), key=lambda span: span[1] - span[0])
    raise ValueError(f"Unknown fusion policy {policy.mode!r}.")


def fit_policy(rows, alternatives):
    def utility(policy):
        return sum(
            temporal_iou(tuple(row["gold"]), fuse(row, alternatives[row["question_id"]], policy))
            for row in rows if row["label"] == 1
        )
    return max(POLICIES, key=utility)


def cross_validate(rows, alternatives, seed):
    tids = {row["transcript_id"] for row in rows}
    folds = grouped_folds(tids, min(5, len(tids)), seed)
    predicted, reports = [], []
    for held_out in folds:
        held = set(held_out)
        train = [row for row in rows if row["transcript_id"] not in held]
        test = [row for row in rows if row["transcript_id"] in held]
        policy = fit_policy(train, alternatives)
        for row in test:
            span = fuse(row, alternatives[row["question_id"]], policy)
            predicted.append({
                **row, "span": span, "original_span": corrected(row),
                "changed": span != corrected(row), "policy": asdict(policy),
            })
        reports.append({
            "train_tids": sorted({row["transcript_id"] for row in train}),
            "held_out_tids": held_out, "policy": asdict(policy),
        })
    return predicted, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--baseline-requests", type=Path, required=True)
    parser.add_argument("--alternative-requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/evidence_fusion"))
    args = parser.parse_args()
    rows = json.loads(args.baseline.read_text())
    other_rows = json.loads(args.alternative.read_text())
    other = {row["question_id"]: row for row in other_rows}
    if len(rows) != 390 or len(other) != 390 or {row["question_id"] for row in rows} != other.keys():
        parser.error("Both complete prediction sets are required.")
    paired_comparison(rows, other_rows, samples=100)
    requests = json.loads(args.baseline_requests.read_text())
    alt_requests = json.loads(args.alternative_requests.read_text())
    left = {row["transcript_id"]: row for row in requests}
    right = {row["transcript_id"]: row for row in alt_requests}
    if left.keys() != right.keys() or any(
        left[tid][key] != right[tid][key]
        for tid in left for key in ("audio_sha256", "transcript_sha256")
    ):
        parser.error("Fusion requires matched audio and transcripts.")
    excluded = {
        tid for request in [*requests, *alt_requests] for tid in request["demonstration_tids"]
    }
    selected = [row for row in rows if row["transcript_id"] not in excluded]
    baseline = [{**row, "span": corrected(row)} for row in selected]
    summary = {
        "excluded_demonstration_tids": sorted(excluded),
        "policies": [asdict(policy) for policy in POLICIES],
        "baseline": score_records(baseline),
        "seeds": {},
        "estimated_combined_max_s": max(
            left[tid]["latency_s"] + right[tid]["latency_s"] for tid in left
        ),
        "latency_note": "Separate-run sum, not an end-to-end HTTP measurement.",
    }
    for seed in (13, 37):
        predictions, folds = cross_validate(selected, other, seed)
        result = {
            "candidate": score_records(predictions),
            "paired": paired_comparison(baseline, predictions, seed=seed),
            "changed_questions": sum(row["changed"] for row in predictions),
            "folds": folds,
        }
        summary["seeds"][str(seed)] = result
        write_json(args.output / f"oof_seed{seed}.json", predictions)
        print(json.dumps({"seed": seed, **{key: value for key, value in result.items() if key != "folds"}}), flush=True)
    summary["qualification"] = "candidate" if all(
        result["paired"]["conversation_bootstrap_95pct"][0] > 0
        for result in summary["seeds"].values()
    ) else "rejected"
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"qualification": summary["qualification"],
                      "estimated_combined_max_s": summary["estimated_combined_max_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
