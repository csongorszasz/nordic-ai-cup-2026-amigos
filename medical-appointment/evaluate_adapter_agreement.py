"""Two fixed agreement rules on complete OOF predictions; no meta-fitting."""

import argparse
import hashlib
import json
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.base import normalize_answer
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


def load_proposals(directory, kind):
    if kind not in ("adapted", "grpo"):
        raise ValueError("Unsupported OOF proposal kind.")
    summary = json.loads((directory / "summary.json").read_text())
    if not summary.get("complete_oof"):
        raise ValueError("Incomplete OOF records cannot qualify an agreement rule.")
    path = directory / f"{kind}_oof.json"
    proposals = json.loads(path.read_text())
    if len(proposals) != summary["questions"] or score_records(proposals) != summary.get(kind):
        raise ValueError("OOF proposals do not match their completed score summary.")
    for row in proposals:
        valid, span = normalize_answer(
            (row["answer"], row["span"]), duration=row.get("duration"), context=row["question_id"],
        )
        if valid != row["answer"] or (list(span) if span is not None else None) != row["span"]:
            raise ValueError("An OOF proposal violates the evidence contract.")
    return summary, proposals, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--proposal-kind", choices=("adapted", "grpo"), default="adapted")
    parser.add_argument("--output", type=Path, default=Path("results/adapter_agreement"))
    args = parser.parse_args()
    summary, proposals, proposal_path = load_proposals(args.oof, args.proposal_kind)
    original = json.loads(args.baseline.read_text())
    excluded = set(summary["excluded_demonstration_tids"])
    rows = [row for row in original if row["transcript_id"] not in excluded]
    baseline = [
        {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row.get("duration")) if row["answer"] else None}
        for row in rows
    ]
    validate_coverage(proposals, rows)
    proposed = {row["question_id"]: row for row in proposals}
    report = {
        "baseline": score_records(baseline), "threshold": 0.5,
        "fit_on_oof": False, "excluded_demonstration_tids": sorted(excluded), "rules": {},
        "proposal_kind": args.proposal_kind,
        "proposal_file": str(proposal_path),
        "proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
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
