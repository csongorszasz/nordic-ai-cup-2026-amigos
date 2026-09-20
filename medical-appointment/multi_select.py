"""Stage 2 of the multi-candidate diagnostic: choose one candidate span and
compare the resulting per-question records against the recorded base arm.

    python multi_select.py --method oracle|s1|s2|s3a

Methods:
  oracle  best candidate by tIoU vs gold (upper bound, not achievable)
  s1      incumbent 26B chooses among the candidates in one prompt
  s2      incumbent 26B mean token log-probability of each candidate
  s3a     Qwen2.5-7B mean token log-probability of each candidate
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

from answerers import modernbert_data as data
from answerers.llm_client import HFClient
from answerers.llm_prompt import qid_for, serialize_transcript
from utils import temporal_iou

PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
_CHOICE_RE = re.compile(r'"choice"\s*:\s*"(c\d+)"', re.IGNORECASE)

JUDGE_SYSTEM = (
    "You extract the single supporting quote from a clinical transcript."
)
CHOICE_SYSTEM = "You select the single best supporting quote for a clinical question."


def judge_messages(transcript, question, candidate):
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"TRANSCRIPT\n{serialize_transcript(transcript)}\n\n"
            f"QUESTION\n{question}\n\n"
            "The supporting quote is:"
        )},
    ]


def choice_messages(transcript, question, quotes):
    listing = "\n".join(f'c{i + 1}: "{quote}"' for i, quote in enumerate(quotes))
    return [
        {"role": "system", "content": CHOICE_SYSTEM},
        {"role": "user", "content": (
            f"TRANSCRIPT\n{serialize_transcript(transcript)}\n\n"
            f"QUESTION\n{question}\n\n"
            f"CANDIDATES\n{listing}\n\n"
            'Which candidate is the exact supporting quote for the question? '
            'Reply with JSON: {"choice":"c1"}'
        )},
    ]


def choose(method, transcript, question, candidates, client):
    """Return the index of the chosen candidate."""
    quotes = candidates["quotes"]
    if len(quotes) == 1:
        return 0
    if method == "oracle":
        return max(range(len(quotes)), key=lambda i: candidates["candidate_ious"][i])
    if method == "s1":
        raw = client.generate(choice_messages(transcript, question, quotes))
        match = _CHOICE_RE.search(raw or "")
        if match is None:
            return 0
        index = int(match.group(1)[1:]) - 1
        return index if 0 <= index < len(quotes) else 0
    scores = [
        client.score_completion(judge_messages(transcript, question, quote), quote)
        for quote in quotes
    ]
    return max(range(len(quotes)), key=lambda i: scores[i])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("oracle", "s1", "s2", "s3a"))
    parser.add_argument("--base", type=Path, default=PROJECT_ROOT / "results" / "llm_p_base_L1_questions.json")
    parser.add_argument("--candidates", type=Path, default=PROJECT_ROOT / "results" / "multi3_candidates.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    base = {row["question_id"]: dict(row) for row in json.loads(args.base.read_text())}
    candidates = {row["question_id"]: row for row in json.loads(args.candidates.read_text())}
    rows_by_tid = defaultdict(list)
    for row in data.load_rows():
        rows_by_tid[row["transcript_id"]].append(row)
    transcripts = {tid: data.load_transcript(tid) for tid in rows_by_tid}
    questions = {
        row["question_id"]: row["question"]
        for rows in rows_by_tid.values() for row in rows
    }

    client = None
    if args.method == "s1":
        client = HFClient(max_new_tokens=64)
    elif args.method == "s2":
        client = HFClient()
    elif args.method == "s3a":
        os.environ.pop("MEDAPP_LLM_REVISION", None)
        client = HFClient(model_name="Qwen/Qwen2.5-7B-Instruct",
                          legacy_special_tokens=False)
    if client is not None:
        client.warm_up()

    changed = 0
    for qid, entry in candidates.items():
        record = base.get(qid)
        if record is None or record.get("prediction") != 1 or not entry["spans"]:
            continue
        transcript = transcripts[entry["transcript_id"]]
        index = choose(args.method, transcript, questions[qid], entry, client)
        record["span"] = list(entry["spans"][index])
        changed += 1

    out = PROJECT_ROOT / "results" / f"multi_{args.method}_questions.json"
    out.write_text(json.dumps(list(base.values()), indent=2))
    print(f"method={args.method} changed_spans={changed} wrote={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
