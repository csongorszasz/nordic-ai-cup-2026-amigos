"""Diagnose the positive-recall bottleneck.

For every annotated positive, score the proposition against three premises:

  1. best single clause          (what the pipeline decides on today)
  2. best clause +/-1 merged     (the cheap "more context" candidate)
  3. the gold evidence text      (capability ceiling: can it recognise the
                                  right evidence when handed it?)

Also scores the hard negatives under premises 1-2 as a precision proxy.

Decision guide:
  * gold high, clause low, merged fixes -> premise construction
  * gold high, merged still low         -> clause selection
  * gold low                            -> capability / paraphrase
  * merged raises hard negatives too    -> precision cost

    NLI_MODEL=... python diagnose_recall.py --tag base
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import asr  # noqa: E402
import answer as answer_module  # noqa: E402
import windows as windows_module  # noqa: E402
from verifier import nli  # noqa: E402
from utils import (  # noqa: E402
    gold_evidence,
    group_questions_by_conversation,
    load_sample_audio,
)

TAU = float(os.environ.get("MEDAPP_NLI_TAU", "0.3"))
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _gold_text(words: List[Dict], gold) -> str:
    tokens = [w["word"] for w in words if w["end"] > gold[0] and w["start"] < gold[1]]
    return " ".join("".join(tokens).split())


def _overlap(question: str, text: str) -> float:
    qtok = answer_module._content_tokens(question)
    if not qtok:
        return 0.0
    ttok = set(_WORD_RE.findall(text.lower()))
    return len(qtok & ttok) / len(qtok)


def _merged_texts(windows) -> List[str]:
    merged = []
    for index in range(len(windows)):
        lo = max(0, index - 1)
        hi = min(len(windows) - 1, index + 1)
        merged.append(" ".join(w.text for w in windows[lo:hi + 1]))
    return merged


def _stats(values: List[float]) -> Dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 3),
        "median": round(statistics.median(values), 3),
        "ge_tau": round(sum(1 for v in values if v >= TAU) / len(values), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()

    print(f"NLI model: {nli.MODEL_NAME}   tau={TAU}")
    nli.warm_up()

    pos = {"best_clause": [], "best_merged": [], "gold_clause": [],
           "gold_merged": [], "gold_text": []}
    neg = {"best_clause": [], "best_merged": []}
    contained_true = {"best_clause": [], "best_merged": []}
    contained_false = {"best_clause": [], "best_merged": []}
    overlaps = []
    n_contained = 0

    for audio_filename, rows in group_questions_by_conversation():
        transcript = asr.transcribe_bytes(
            load_sample_audio(audio_filename), audio_filename, cache=True
        )
        words = transcript.get("words", [])
        windows = windows_module.windows_from_transcript(transcript)
        if not windows:
            continue
        clause_texts = [w.text for w in windows]
        merged_texts = _merged_texts(windows)

        for row in rows:
            question = row["question"]
            hypothesis = answer_module.hypothesis_for(question)
            clause_scores = nli.score(clause_texts, hypothesis)
            merged_scores = nli.score(merged_texts, hypothesis)

            gold = gold_evidence(row)
            if gold is None:
                neg["best_clause"].append(max(clause_scores))
                neg["best_merged"].append(max(merged_scores))
                continue

            best_clause = max(clause_scores)
            best_merged = max(merged_scores)

            overlapping = [
                i for i, w in enumerate(windows)
                if w.end > gold[0] and w.start < gold[1]
            ]
            gold_clause = max((clause_scores[i] for i in overlapping), default=0.0)
            gold_merged = max((merged_scores[i] for i in overlapping), default=0.0)
            gold_text = _gold_text(words, gold)
            gold_score = nli.score([gold_text], hypothesis)[0] if gold_text else 0.0

            contained = any(
                w.start <= gold[0] and w.end >= gold[1] for w in windows
            )
            n_contained += int(contained)

            pos["best_clause"].append(best_clause)
            pos["best_merged"].append(best_merged)
            pos["gold_clause"].append(gold_clause)
            pos["gold_merged"].append(gold_merged)
            pos["gold_text"].append(gold_score)
            overlaps.append(_overlap(question, gold_text))
            bucket = contained_true if contained else contained_false
            bucket["best_clause"].append(best_clause)
            bucket["best_merged"].append(best_merged)

    summary = {
        "tag": args.tag,
        "model": nli.MODEL_NAME,
        "tau": TAU,
        "positives": {k: _stats(v) for k, v in pos.items()},
        "hard_negatives": {k: _stats(v) for k, v in neg.items()},
        "contained_true": {k: _stats(v) for k, v in contained_true.items()},
        "contained_false": {k: _stats(v) for k, v in contained_false.items()},
        "contained_share": round(n_contained / max(1, len(pos["best_clause"])), 3),
        "overlap_mean": round(sum(overlaps) / max(1, len(overlaps)), 3),
    }

    print("\n=== positives (n=%d) ===" % len(pos["best_clause"]))
    for key, stats in summary["positives"].items():
        print(f"  {key:<14} mean {stats['mean']:.3f}  median {stats['median']:.3f}  "
              f">=tau {stats['ge_tau']:.0%}")
    print("\n=== hard negatives (precision proxy, n=%d) ===" % len(neg["best_clause"]))
    for key, stats in summary["hard_negatives"].items():
        print(f"  {key:<14} mean {stats['mean']:.3f}  >=tau {stats['ge_tau']:.0%}")
    print(f"\ngold contained in one clause: {summary['contained_share']:.0%}")
    print("\n=== by containment ===")
    for label in ("contained_false", "contained_true"):
        s = summary[label]
        print(f"  {label:<16} best_clause mean {s['best_clause'].get('mean',0):.3f}"
              f"  best_merged mean {s['best_merged'].get('mean',0):.3f}")

    out = PROJECT_ROOT / "results" / f"diag_recall_{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
