"""Audit overlap-derived versus inference-aware tIoU-optimal training targets."""

import argparse
import json
import logging
from pathlib import Path

from answerers.align import align_quote_matches, text_between
from answerers.boundaries import adjusted_span
from answerers.span_targets import optimal_quote_target
from benchmark import write_json
from utils import temporal_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/quote_target_audit"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rows = json.loads((args.baseline / "base_legacy_questions.json").read_text())
    requests = json.loads((args.baseline / "base_legacy_conversations.json").read_text())
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    transcripts = {}
    records = []
    for row in rows:
        if row["label"] != 1:
            continue
        tid = row["transcript_id"]
        if tid not in transcripts:
            transcripts[tid] = json.loads((args.baseline / "transcripts" / f"{tid}.json").read_text())
        transcript = transcripts[tid]
        words, gold, duration = transcript["words"], row["gold"], transcript["duration"]
        overlap_quote = text_between(words, *gold)
        matches = align_quote_matches(words, overlap_quote)
        decodable = len(matches) == 1
        overlap_span = adjusted_span(matches[0][:2], (0.2, 0.0), duration) if decodable else None
        optimal = optimal_quote_target(words, gold, duration, unique=True)
        unconstrained = optimal_quote_target(words, gold, duration, unique=False)
        if optimal is None:
            logging.warning(
                "%s has no uniquely decodable target overlapping gold under the fixed correction.",
                row["question_id"],
            )
        records.append({
            "question_id": row["question_id"], "transcript_id": tid,
            "overlap_quote": overlap_quote, "overlap_decodable": decodable,
            "overlap_tiou": temporal_iou(tuple(gold), overlap_span),
            "optimal": optimal,
            "optimal_tiou": optimal["tiou"] if optimal is not None else 0.0,
            "unconstrained_tiou": unconstrained["tiou"] if unconstrained is not None else 0.0,
            "text_changed": optimal is not None and optimal["quote"] != overlap_quote,
        })
    safe = [row for row in records if row["transcript_id"] not in excluded]
    def summarize(items):
        return {
            "positives": len(items),
            "changed_targets": sum(row["text_changed"] for row in items),
            "undecodable_overlap_targets": sum(not row["overlap_decodable"] for row in items),
            "no_feasible_unique_target": sum(row["optimal"] is None for row in items),
            "mean_overlap_tiou": sum(row["overlap_tiou"] for row in items) / len(items),
            "mean_optimal_unique_tiou": sum(row["optimal_tiou"] for row in items) / len(items),
            "mean_unconstrained_tiou": sum(row["unconstrained_tiou"] for row in items) / len(items),
        }
    summary = {
        "all": summarize(records), "non_demo": summarize(safe),
        "excluded_demonstration_tids": sorted(excluded),
        "diagnostic_only": True, "reference_annotations_modified": False,
    }
    write_json(args.output / "targets.json", records)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
