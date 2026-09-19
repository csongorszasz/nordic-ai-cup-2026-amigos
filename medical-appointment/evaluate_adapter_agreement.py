"""Two fixed agreement rules on complete OOF predictions; no meta-fitting."""

import argparse
import json
from pathlib import Path

from answerers.boundaries import adjusted_span
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from train_quote_oof import validate_coverage
from utils import temporal_iou


def agreed_span(baseline, proposed, mode):
    if mode not in ("adapter_on_agreement", "midpoint_on_agreement"):
        raise ValueError("Unknown fixed agreement rule.")
    if baseline is None or proposed is None:
        return baseline
    if temporal_iou(tuple(baseline), tuple(proposed)) < 0.5:
        return baseline
    if mode == "adapter_on_agreement":
        return list(proposed)
    return [(left + right) / 2 for left, right in zip(baseline, proposed)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/adapter_agreement"))
    args = parser.parse_args()
    summary = json.loads((args.oof / "summary.json").read_text())
    if not summary.get("complete_oof"):
        raise ValueError("Incomplete OOF records cannot qualify an agreement rule.")
    original = json.loads(args.baseline.read_text())
    excluded = set(summary["excluded_demonstration_tids"])
    rows = [row for row in original if row["transcript_id"] not in excluded]
    baseline = [
        {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row.get("duration")) if row["answer"] else None}
        for row in rows
    ]
    proposals = json.loads((args.oof / "adapted_oof.json").read_text())
    validate_coverage(proposals, rows)
    proposed = {row["question_id"]: row for row in proposals}
    report = {
        "baseline": score_records(baseline), "threshold": 0.5,
        "fit_on_oof": False, "excluded_demonstration_tids": sorted(excluded), "rules": {},
    }
    for mode in ("adapter_on_agreement", "midpoint_on_agreement"):
        records = []
        for row in baseline:
            span = agreed_span(row["span"], proposed[row["question_id"]]["span"], mode)
            records.append({**row, "span": span, "changed": span != row["span"]})
        validate_coverage(records, rows)
        result = {
            "candidate": score_records(records),
            "paired": paired_comparison(baseline, records),
            "changed_questions": sum(row["changed"] for row in records),
        }
        report["rules"][mode] = result
        write_json(args.output / f"{mode}.json", records)
        print(json.dumps({"rule": mode, **result}), flush=True)
    write_json(args.output / "summary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
