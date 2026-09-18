"""NLI-validate the annotated quotes.

A support quote should entail the question's proposition; a refute quote should
NOT (it states the different, actual value). Writes annotations/nli_check.csv
and prints a summary so suspicious rows can be reviewed.

    python annotations/nli_check.py
"""

import csv
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from questions import to_proposition  # noqa: E402
from verifier import nli  # noqa: E402

TAU = 0.3
EVIDENCE = ROOT / "annotations" / "evidence.csv"
OUT = ROOT / "annotations" / "nli_check.csv"


def main() -> int:
    rows = list(csv.DictReader(open(EVIDENCE)))
    questions = {
        r["question_id"]: r["question"]
        for r in csv.DictReader(open(ROOT / "data" / "question_train.csv"))
    }
    nli.warm_up()

    out: List[dict] = []
    for row in rows:
        quote = row["quote"]
        if not quote:
            continue
        question = questions.get(row["question_id"], "")
        hypothesis = to_proposition(question)
        score = nli.score([quote], hypothesis)[0]
        bucket = row["bucket"]
        if bucket == "refute":
            verdict = "ok" if score < TAU else "SUSPICIOUS_entails"
        elif bucket == "support":
            verdict = "ok" if score >= TAU else "WEAK"
        else:
            verdict = "n/a"
        out.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "bucket": bucket,
                "entailment": round(score, 3),
                "verdict": verdict,
                "question": question,
                "quote": quote,
            }
        )

    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    refutes = [r for r in out if r["bucket"] == "refute"]
    supports = [r for r in out if r["bucket"] == "support"]
    susp = [r for r in refutes if r["verdict"] != "ok"]
    weak = [r for r in supports if r["verdict"] != "ok"]
    print(f"wrote {OUT}")
    print(f"refutes {len(refutes)}: mean entailment "
          f"{sum(r['entailment'] for r in refutes)/len(refutes):.3f}; "
          f"suspicious (>= {TAU}) {len(susp)}")
    for r in susp:
        print(f"  SUSPICIOUS refute {r['question_id']} entailment {r['entailment']}")
    print(f"supports {len(supports)}: mean entailment "
          f"{sum(r['entailment'] for r in supports)/len(supports):.3f}; "
          f"weak (< {TAU}) {len(weak)}")
    for r in weak[:10]:
        print(f"  WEAK support {r['question_id']} entailment {r['entailment']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
