"""Per-question citation-confidence diagnostic for the L1 LLM.

Development-only. Runs the incumbent L1 prompt once per conversation, captures
the per-token log-probability of the generated JSON, attributes it to each
question's entry, and reports whether confidence separates localization hits
from misses. Writes ``results/confidence_<tag>_questions.json``.

    python confidence_probe.py --limit 3          # smoke
    python confidence_probe.py                    # full 39
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from answerers import modernbert_data as data
from answerers.align import align_span
from answerers.base import normalize_answer
from answerers.llm_client import HFClient
from answerers.llm_parse import parse_answers
from answerers.llm_prompt import build_few_shot, build_l1_messages, qid_for
from utils import temporal_iou

PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
_ID_RE = re.compile(r'"id"\s*:\s*"(q\d+)"')


def segment_confidence(tokenizer, token_ids, logprobs, text):
    """Mean log-probability of each question's JSON entry in the completion."""
    if not token_ids or not text:
        return {}
    boundaries = [len(tokenizer.decode(token_ids[:k], skip_special_tokens=True))
                  for k in range(len(token_ids) + 1)]
    matches = list(_ID_RE.finditer(text))
    result = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        indices = [k for k in range(len(token_ids))
                   if boundaries[k + 1] > start and boundaries[k] < end]
        if indices:
            result[match.group(1)] = float(np.mean([logprobs[k] for k in indices]))
    return result


def separation(hits, misses):
    """P(hit confidence > miss confidence) with ties counted as a half."""
    if not hits or not misses:
        return None
    a = np.asarray(hits)[:, None]
    b = np.asarray(misses)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def summarize(records):
    answered = [r for r in records if r["answer"] and r["confidence"] is not None]
    hits = [r["confidence"] for r in answered if r["iou"] >= 0.5]
    misses = [r["confidence"] for r in answered if r["iou"] < 0.5]
    print(f"positives={len(records)} answered={len(answered)} hits={len(hits)} misses={len(misses)}")
    for name, values in (("hits", hits), ("misses", misses)):
        if values:
            arr = np.asarray(values)
            print(f"  {name:<7} conf median={np.median(arr):+.3f} "
                  f"q25={np.quantile(arr, .25):+.3f} q75={np.quantile(arr, .75):+.3f}")
    auc = separation(hits, misses)
    if auc is not None:
        print(f"  separation P(hit>miss) = {auc:.3f}  (0.5 = no signal)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="L1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    rows = data.load_rows()
    rows_by_tid = defaultdict(list)
    for row in rows:
        rows_by_tid[row["transcript_id"]].append(row)
    transcripts = {tid: data.load_transcript(tid) for tid in rows_by_tid}
    evidence = data.load_evidence()
    conversations = list(rows_by_tid.items())
    if args.limit:
        conversations = conversations[:args.limit]

    client = HFClient(model_name=args.model, max_new_tokens=args.max_new_tokens)
    client.warm_up()

    records = []
    for index, (tid, conv_rows) in enumerate(conversations):
        transcript = transcripts[tid]
        words = transcript.get("words", [])
        questions = [row["question"] for row in conv_rows]
        ids = [qid_for(i) for i in range(len(questions))]
        few_shot = build_few_shot(rows_by_tid, transcripts, evidence, exclude_tid=tid)
        started = time.perf_counter()
        raw, logprobs, token_ids = client.generate_scored(
            build_l1_messages(transcript, questions, few_shot)
        )
        elapsed = time.perf_counter() - started
        per_question = segment_confidence(client._tokenizer, token_ids, logprobs, raw)
        parsed = parse_answers(raw, ids)
        for qid, row in zip(ids, conv_rows):
            if int(row["label"]) != 1:
                continue
            entry = parsed.get(qid) or {}
            answer = entry.get("answer")
            quote = entry.get("quote")
            span = align_span(words, quote or "") if answer else None
            _, span = normalize_answer(
                (answer is True, span), duration=transcript.get("duration"),
                context=row["question_id"],
            )
            gold = [float(row["evidence_start"]), float(row["evidence_end"])]
            iou = temporal_iou(span, gold) if (answer and span) else 0.0
            records.append(
                {
                    "question_id": row["question_id"],
                    "transcript_id": tid,
                    "answer": bool(answer),
                    "quote": quote,
                    "span": list(span) if span else None,
                    "gold": gold,
                    "iou": round(float(iou), 4),
                    "confidence": per_question.get(qid),
                    "tokens": len(token_ids),
                }
            )
        print(f"  [{index + 1}/{len(conversations)}] {tid} {elapsed:5.1f}s "
              f"mapped={len(per_question)}/{len(ids)}")

    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    (PROJECT_ROOT / "results" / f"confidence_{args.tag}_questions.json").write_text(
        json.dumps(records, indent=2)
    )
    summarize(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
