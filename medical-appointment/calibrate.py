"""Calibrate the decision threshold on out-of-fold (OOF) records, LOCO-style.

Reads the records written by ``dev_eval.py --oof`` (generated once with
``MEDAPP_NLI_TAU=0`` so every question is localized). Derives a prediction for
any threshold from the stored ``p`` and ``span``, so no re-inference is needed.

    MEDAPP_NLI_TAU=0 python dev_eval.py --answer nli --oof results/oof.json
    python calibrate.py results/oof.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import temporal_iou  # noqa: E402

GRID = [round(0.05 * i, 2) for i in range(20)]  # 0.00 .. 0.95


def predict(record: Dict, tau: float) -> Tuple[bool, Optional[Tuple[float, float]]]:
    p = record.get("p")
    passes = p is not None and (
        p >= tau if record.get("threshold_inclusive", True) else p > tau
    )
    ok = bool(record.get("guard_ok", True)) and passes
    proposed = record.get("proposed_span", record.get("span"))
    span = tuple(proposed) if (ok and proposed) else None
    return ok, span


def evaluate(records: List[Dict], tau: float) -> Tuple[float, float, float]:
    correct = 0
    tious: List[float] = []
    for record in records:
        pred, span = predict(record, tau)
        correct += int(pred == bool(record["label"]))
        if record["label"] == 1 and record.get("gold") is not None:
            tious.append(temporal_iou(tuple(record["gold"]), span))
    accuracy = correct / len(records) if records else 0.0
    mean_tiou = sum(tious) / len(tious) if tious else 0.0
    return 0.4 * accuracy + 0.6 * mean_tiou, accuracy, mean_tiou


def best_tau(records: List[Dict]) -> float:
    return max(GRID, key=lambda tau: evaluate(records, tau)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="OOF records JSON from dev_eval --oof")
    args = parser.parse_args()

    records = json.loads(Path(args.path).read_text())
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        groups[record["transcript_id"]].append(record)

    print(f"records {len(records)}  conversations {len(groups)}")

    # In-sample best (optimistic; for reference only).
    global_tau = best_tau(records)
    gscore, gacc, giou = evaluate(records, global_tau)
    print(f"global best tau {global_tau:.2f} (in-sample, optimistic): "
          f"score {gscore:.3f}  acc {gacc:.3f}  mIoU {giou:.3f}")

    # LOCO: choose tau on the other conversations, apply to the held-out one.
    pooled: List[Tuple[Dict, bool, Optional[Tuple[float, float]]]] = []
    taus: List[float] = []
    for held_out, held_records in groups.items():
        train = [r for r in records if r["transcript_id"] != held_out]
        tau = best_tau(train)
        taus.append(tau)
        for record in held_records:
            pred, span = predict(record, tau)
            pooled.append((record, pred, span))

    correct = sum(int(pred == bool(r["label"])) for r, pred, _ in pooled)
    accuracy = correct / len(pooled)
    tious = [
        temporal_iou(tuple(r["gold"]), span)
        for r, pred, span in pooled
        if r["label"] == 1 and r.get("gold") is not None
    ]
    mean_tiou = sum(tious) / len(tious) if tious else 0.0
    loco_score = 0.4 * accuracy + 0.6 * mean_tiou
    print(f"LOCO median tau {sorted(taus)[len(taus)//2]:.2f}  "
          f"(min {min(taus):.2f} max {max(taus):.2f})")
    print(f"LOCO pooled: score {loco_score:.3f}  acc {accuracy:.3f}  "
          f"mIoU {mean_tiou:.3f}")

    # Score-aware threshold: a missed positive loses accuracy *and* tIoU, so the
    # optimal cutoff is below 0.5. Derivation (accuracy over all questions, tIoU
    # over positives only): yes iff p > 0.4 / (0.8 + 1.2 q).
    answered_yes = [
        (r, span) for r, pred, span in pooled
        if pred and r["label"] == 1 and r.get("gold") is not None
    ]
    q = (
        sum(temporal_iou(tuple(r["gold"]), span) for r, span in answered_yes)
        / len(answered_yes)
        if answered_yes
        else 0.0
    )
    if q > 0:
        aware = 0.4 / (0.8 + 1.2 * q)
        print(f"mean tIoU when answered yes q={q:.3f} -> score-aware tau={aware:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
