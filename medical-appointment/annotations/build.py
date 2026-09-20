"""Merge the agent's annotation drafts into one reviewed evidence table.

Positives keep the supplied gold spans (the agent's positive quotes are a
different-but-valid convention, so they are not used as labels). Hard negatives
use the agent's refute/absent decision, with the quote aligned back to a word
span for timestamps. Off-topic is absent.

    python annotations/build.py
    -> annotations/evidence.csv
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
DRAFTS = ROOT / "annotations" / "drafts" / "ALL.json"
QUESTIONS = ROOT / "data" / "question_train.csv"
OOF = ROOT / "results" / "oof_T030.json"
OUT = ROOT / "annotations" / "evidence.csv"


def load_words(transcript_id: str):
    path = TRANSCRIPTS / f"conversation_{transcript_id}.dc5ba020.json"
    if not path.exists():
        path = sorted(TRANSCRIPTS.glob(f"conversation_{transcript_id}.*.json"))[0]
    return json.loads(path.read_text())["words"]


def align(words, quote):
    """Verbatim quote -> (start, end, first_word, last_word), or None."""
    text = "".join(w["word"] for w in words)
    char_to_word = []
    for index, word in enumerate(words):
        char_to_word.extend([index] * len(word["word"]))
    tokens = [re.escape(t) for t in quote.split()]
    match = re.compile(r"\s+".join(tokens), re.IGNORECASE).search(text)
    if not match:
        return None
    i, j = char_to_word[match.start()], char_to_word[match.end() - 1]
    return words[i]["start"], words[j]["end"], i, j


def text_between(words, start, end):
    tokens = [w["word"] for w in words if w["end"] > start and w["start"] < end]
    return " ".join("".join(tokens).split())


def main() -> int:
    rows = list(csv.DictReader(open(QUESTIONS)))
    drafts = {r["question_id"]: r for r in json.loads(DRAFTS.read_text())}
    oof = {}
    if OOF.exists():
        oof = {r["question_id"]: r for r in json.loads(OOF.read_text())}

    words_cache = {}
    out = []
    for row in rows:
        qid = row["question_id"]
        tid = row["transcript_id"]
        qtype = row["question_type"]
        draft = drafts.get(qid, {})
        words = words_cache.setdefault(tid, load_words(tid))

        flags = []
        if draft.get("confidence") == "low":
            flags.append("low_confidence")
        p = (oof.get(qid) or {}).get("p")

        if qtype == "positive":
            start, end = float(row["evidence_start"]), float(row["evidence_end"])
            quote = text_between(words, start, end)
            bucket, source = "support", "gold"
            if p is not None and p < 0.3:
                flags.append("nli_miss")
        elif qtype == "off_topic":
            bucket, quote, start, end, source = "absent", "", "", "", "n/a"
        else:  # hard_negative
            bucket = draft.get("bucket", "absent")
            quote = draft.get("quote", "") if bucket == "refute" else ""
            span = align(words, quote) if quote else None
            if bucket == "refute" and span is None:
                flags.append("alignment_failed")
                bucket, quote = "absent", ""
                start, end = "", ""
            else:
                start = round(span[0], 2) if span else ""
                end = round(span[1], 2) if span else ""
            source = "agent"
            if bucket == "refute" and p is not None and p >= 0.5:
                flags.append("nli_error")

        out.append(
            {
                "question_id": qid,
                "transcript_id": tid,
                "question_type": qtype,
                "bucket": bucket,
                "quote": quote,
                "start": start,
                "end": end,
                "quote_source": source,
                "confidence": draft.get("confidence", ""),
                "nli_p": round(p, 3) if p is not None else "",
                "flags": "|".join(flags),
            }
        )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {OUT} ({len(out)} rows)")
    from collections import Counter

    print("buckets:", dict(Counter(r["bucket"] for r in out)))
    flagged = [r for r in out if r["flags"]]
    print(f"flagged rows: {len(flagged)}")
    for r in flagged[:20]:
        print(f"  {r['question_id']} [{r['question_type']}] {r['flags']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
