"""Build annotations/review.md: the rows a human should check before training.

Categories:
  1. refutes whose quote entails the question (possible wrong bucket)
  2. refutes that are current NLI errors (the high-value near-miss rationales)
  3. low-confidence hard negatives marked absent (confirm truly absent)
  4. low-confidence off-topic marked absent
  5. positives the NLI misses (confirm the gold span)

    python annotations/review.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "annotations" / "evidence.csv"
CHECK = ROOT / "annotations" / "nli_check.csv"
OUT = ROOT / "annotations" / "review.md"


def main() -> int:
    ev = {r["question_id"]: r for r in csv.DictReader(open(EVIDENCE))}
    check = {r["question_id"]: r for r in csv.DictReader(open(CHECK))}
    questions = {
        r["question_id"]: r
        for r in csv.DictReader(open(ROOT / "data" / "question_train.csv"))
    }

    lines = ["# Annotation review\n",
             "Edit the `decision` column (keep/change-to-absent/change-to-refute).\n"]

    def section(title, rows):
        lines.append(f"\n## {title} ({len(rows)})\n")
        lines.append("| question_id | question | quote | note | decision |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            e = ev[r["question_id"]]
            q = questions[r["question_id"]]["question"].replace("|", "/")
            quote = e["quote"].replace("|", "/")
            lines.append(
                f"| {r['question_id']} | {q} | {quote} | {r['note']} | keep |"
            )

    susp = [
        {"question_id": c["question_id"], "note": f"entailment {c['entailment']}"}
        for c in check.values()
        if c["verdict"] == "SUSPICIOUS_entails"
    ]
    section("Refutes whose quote entails (check bucket)", susp)

    errs = [
        {"question_id": e["question_id"],
         "note": f"NLI error, entailment {check.get(e['question_id'],{}).get('entailment','?')}"}
        for e in ev.values()
        if "nli_error" in e["flags"]
    ]
    section("Refutes that fix an NLI error (confirm near-miss)", errs)

    low_hn = [
        {"question_id": e["question_id"], "note": "low confidence, absent"}
        for e in ev.values()
        if e["bucket"] == "absent" and e["question_type"] == "hard_negative"
        and e["confidence"] == "low"
    ]
    section("Low-confidence hard negatives marked absent", low_hn)

    low_ot = [
        {"question_id": e["question_id"], "note": "low confidence, off-topic absent"}
        for e in ev.values()
        if e["question_type"] == "off_topic" and e["confidence"] == "low"
    ]
    section("Low-confidence off-topic", low_ot)

    misses = [
        {"question_id": e["question_id"], "note": f"NLI missed (p={e['nli_p']})"}
        for e in ev.values()
        if "nli_miss" in e["flags"]
    ]
    section("Positives the NLI misses (confirm gold span)", misses)

    OUT.write_text("\n".join(lines) + "\n")
    total = len(susp) + len(errs) + len(low_hn) + len(low_ot) + len(misses)
    print(f"wrote {OUT} ({total} rows to review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
