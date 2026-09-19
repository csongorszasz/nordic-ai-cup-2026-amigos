"""Disjoint-occurrence subset comparison for prompt arms.

Development-only. Scores only the positives the recorded base arm failed
disjointly (tIoU == 0), so the subset is oracle-selected: this is a mechanism
diagnostic, not a deployment estimate.

    python compare_subset.py \
        --base results/llm_p2_base_L1_questions.json \
        --arm results/llm_dis_v1_L1_questions.json \
        --arm results/llm_dis_v1r_L1_questions.json
"""

import argparse
import json
from pathlib import Path

from answerers.boundaries import adjusted_span


def iou(a, b):
    if not a or not b:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def disjoint_qids(base):
    return sorted(
        row["question_id"] for row in base
        if row.get("question_type") == "positive"
        and iou(row.get("span"), row.get("gold")) == 0.0
    )


def score_subset(records, qids, offset=None):
    by_qid = {row["question_id"]: row for row in records}
    correct = 0
    tious = []
    for qid in qids:
        row = by_qid.get(qid)
        if row is None:
            raise ValueError(f"{qid} is missing from the records.")
        span = row.get("span")
        if offset is not None and span:
            span = list(adjusted_span(span, offset, None))
        correct += 1 if row.get("prediction") == 1 else 0
        tious.append(iou(span, row.get("gold")))
    count = len(qids)
    accuracy = correct / count
    mean_tiou = sum(tious) / count
    return {
        "score": 0.4 * accuracy + 0.6 * mean_tiou,
        "accuracy": accuracy,
        "mean_tiou": mean_tiou,
        "tious": tious,
    }


def report(label, stats):
    print(
        f"  {label:<26} score {stats['score']:.4f}  "
        f"mIoU {stats['mean_tiou']:.4f}  acc {stats['accuracy']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", type=Path, action="append", required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    qids = disjoint_qids(base)
    print(f"disjoint positives (base tIoU==0): {len(qids)}")
    print("base:")
    report("raw", score_subset(base, qids))
    report("fixed+0.2", score_subset(base, qids, (0.2, 0.0)))

    for path in args.arm:
        arm = json.loads(path.read_text())
        print(f"{path.stem}:")
        raw = score_subset(arm, qids)
        fixed = score_subset(arm, qids, (0.2, 0.0))
        report("raw", raw)
        report("fixed+0.2", fixed)
        base_by_qid = {row["question_id"]: row for row in base}
        improved = [
            qid for qid, value in zip(qids, fixed["tious"])
            if value > iou(base_by_qid[qid].get("span"), base_by_qid[qid].get("gold")) + 1e-9
        ]
        worsened = [
            qid for qid, value in zip(qids, fixed["tious"])
            if value < iou(base_by_qid[qid].get("span"), base_by_qid[qid].get("gold")) - 1e-9
        ]
        print(f"  recovered {len(improved)}: {improved}")
        print(f"  worsened  {len(worsened)}: {worsened}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
