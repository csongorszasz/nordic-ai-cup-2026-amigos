"""Fit span boundary offsets to the annotators' convention.

Reads the per-question records of an ``llm_probe.py --rungs SERVED`` run (made
with zero offsets) and grid-searches a start and an end offset, in seconds,
that maximise mean tIoU over the annotated positives. Reports the fit
leave-one-conversation-out too, so an offset that only fits noise shows up as
no held-out gain.

    python calibrate_spans.py results/llm_base_SERVED_questions.json

Apply the result with ``MEDAPP_SPAN_START_OFFSET`` / ``MEDAPP_SPAN_END_OFFSET``
(or ``llm_probe.py --offsets`` to confirm it end to end).
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from answerers.align import apply_offsets
from utils import temporal_iou

GRID = [round(-1.0 + 0.05 * step, 2) for step in range(41)]  # -1.00 .. +1.00 s


def _mean_tiou(records: Sequence[Dict], start: float, end: float) -> float:
    scores = []
    for record in records:
        span = record["span"]
        if not record["answer"] or span is None:
            scores.append(0.0)
            continue
        shifted = apply_offsets(tuple(span), start, end)
        scores.append(temporal_iou(tuple(record["gold"]), shifted))
    return sum(scores) / len(scores) if scores else 0.0


def fit(records: Sequence[Dict]) -> Tuple[float, float, float]:
    """Best ``(start_offset, end_offset, mean_tiou)`` on ``records``."""
    best = (0.0, 0.0, _mean_tiou(records, 0.0, 0.0))
    for start in GRID:
        for end in GRID:
            score = _mean_tiou(records, start, end)
            if score > best[2] + 1e-9:
                best = (start, end, score)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions_json")
    args = parser.parse_args()

    records = [
        r for r in json.loads(open(args.questions_json).read())
        if r.get("gold")
    ]
    by_tid: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        by_tid[record["transcript_id"]].append(record)

    base = _mean_tiou(records, 0.0, 0.0)
    start, end, fitted = fit(records)

    held_out: List[float] = []
    for tid, rows in by_tid.items():
        train = [r for r in records if r["transcript_id"] != tid]
        s, e, _ = fit(train)
        held_out.extend([_mean_tiou([r], s, e) for r in rows])
    loco = sum(held_out) / len(held_out) if held_out else 0.0

    deltas = [
        (r["span"][0] - r["gold"][0], r["span"][1] - r["gold"][1])
        for r in records if r["answer"] and r["span"]
    ]
    if deltas:
        starts = sorted(d[0] for d in deltas)
        ends = sorted(d[1] for d in deltas)
        print(f"median start delta (pred-gold) {starts[len(starts) // 2]:+.2f}s   "
              f"median end delta {ends[len(ends) // 2]:+.2f}s   (n={len(deltas)})")
    print(f"mean tIoU  no offset {base:.4f}")
    print(f"           fitted    {fitted:.4f}   start {start:+.2f}s  end {end:+.2f}s")
    print(f"           LOCO      {loco:.4f}   (held-out; the number to trust)")
    print(f"score delta ~ {0.6 * (loco - base):+.4f}")
    print(f"\nexport MEDAPP_SPAN_START_OFFSET={start}\nexport MEDAPP_SPAN_END_OFFSET={end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
