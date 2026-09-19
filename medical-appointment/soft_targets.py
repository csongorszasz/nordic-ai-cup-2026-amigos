"""Soft-target / multi-annotator analysis for evidence spans.

For each positive question, collect the valid spans from three annotators:
the official gold, the independent agent draft, and the incumbent model. Report
span disagreement and how a consensus (soft) target compares with the hard gold
that the competition actually scores.

    python soft_targets.py
"""

import csv
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from answerers.align import align_quote_matches

PROJECT_ROOT = Path(__file__).resolve().parent


def _load(path):
    return json.loads(Path(path).read_text())


def _span(tid, quote, words_cache):
    if not quote:
        return None
    words = words_cache.setdefault(tid, _words(tid))
    matches = align_quote_matches(words, quote)
    return [round(matches[0][0], 3), round(matches[0][1], 3)] if matches else None


def _words(tid):
    for path in (
        PROJECT_ROOT / "transcripts" / f"conversation_{tid}.e75a7f6e.json",
        PROJECT_ROOT / "transcripts" / f"conversation_{tid}.dc5ba020.json",
    ):
        if path.exists():
            return json.loads(path.read_text())["words"]
    matches = sorted((PROJECT_ROOT / "transcripts").glob(f"conversation_{tid}.*.json"))
    return json.loads(matches[0].read_text())["words"]


def iou(a, b):
    if not a or not b:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def medoid(spans):
    spans = [s for s in spans if s]
    if not spans:
        return None
    return max(spans, key=lambda s: sum(iou(s, other) for other in spans))


def main() -> int:
    base = {r["question_id"]: r for r in _load(PROJECT_ROOT / "results" / "llm_p_base_L1_questions.json")} \
        if (PROJECT_ROOT / "results" / "llm_p_base_L1_questions.json").exists() else \
        {r["question_id"]: r for r in _load("/tmp/opencode/conf.json")}
    evidence = {r["question_id"]: r for r in csv.DictReader(open(PROJECT_ROOT / "annotations" / "evidence.csv"))}
    drafts = {r["question_id"]: r for r in _load(PROJECT_ROOT / "annotations" / "drafts" / "ALL.json")}
    words_cache = {}

    rows = []
    for qid, record in base.items():
        if record.get("label", 1) != 1 or not record.get("answer") or not record.get("span"):
            continue
        gold = record.get("gold") or [
            float(evidence[qid]["start"]), float(evidence[qid]["end"])
        ] if evidence.get(qid, {}).get("start") else None
        agent = _span(record["transcript_id"], drafts.get(qid, {}).get("quote"), words_cache)
        model = record["span"]
        rows.append({"question_id": qid, "gold": gold, "agent": agent, "model": model})

    n = len(rows)
    all_three = [r for r in rows if r["gold"] and r["agent"] and r["model"]]
    print(f"positives with model span: {n}; all three annotators: {len(all_three)}")

    def mean_iou(a_key, b_key, subset):
        return statistics.mean(iou(r[a_key], r[b_key]) for r in subset)

    subset = all_three or rows
    print(f"mean tIoU model-gold   : {mean_iou('model', 'gold', subset):.4f}")
    print(f"mean tIoU agent-gold   : {mean_iou('agent', 'gold', subset):.4f}")
    print(f"mean tIoU model-agent  : {mean_iou('model', 'agent', subset):.4f}")

    pairwise = [iou(r["gold"], r["agent"]) for r in all_three] + \
               [iou(r["gold"], r["model"]) for r in all_three] + \
               [iou(r["agent"], r["model"]) for r in all_three]
    print(f"mean pairwise annotator IoU: {statistics.mean(pairwise):.4f} "
          f"(median {statistics.median(pairwise):.4f})")

    medoid_iou = [iou(medoid([r["gold"], r["agent"], r["model"]]), r["gold"]) for r in all_three]
    best_of_three = [max(iou(r["gold"], r["gold"]), iou(r["agent"], r["gold"]), iou(r["model"], r["gold"])) for r in all_three]
    print(f"soft-consensus (medoid) vs gold: {statistics.mean(medoid_iou):.4f}")
    print(f"oracle-best-of-3 vs gold       : {statistics.mean(best_of_three):.4f}")

    disagree = [1 - statistics.mean([iou(r["gold"], r["agent"]), iou(r["gold"], r["model"]), iou(r["agent"], r["model"])]) for r in all_three]
    print(f"mean per-question disagreement (1 - mean pairwise IoU): {statistics.mean(disagree):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
