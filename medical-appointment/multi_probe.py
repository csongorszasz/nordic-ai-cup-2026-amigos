"""Stage 1 of the multi-candidate diagnostic: generate up to three supporting
quotes per yes with the incumbent 26B and measure the candidate oracle.

    python multi_probe.py --limit 39
"""

import argparse
import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path

from answerers import modernbert_data as data
from answerers.align import align_span
from answerers.llm_client import HFClient
from answerers.llm_parse import parse_quotes
from answerers.llm_prompt import build_few_shot, build_l1_messages, qid_for
from utils import temporal_iou

PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="multi3")
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
        raw = client.generate(build_l1_messages(transcript, questions, few_shot, variant="multi3"))
        elapsed = time.perf_counter() - started
        parsed = parse_quotes(raw, ids)
        conversation_records = []
        for qid, row in zip(ids, conv_rows):
            if int(row["label"]) != 1:
                continue
            entry = parsed.get(qid) or {}
            spans, kept = [], []
            for quote in entry.get("quotes") or []:
                span = align_span(words, quote)
                if span is not None:
                    spans.append([round(span[0], 3), round(span[1], 3)])
                    kept.append(quote)
            gold = [float(row["evidence_start"]), float(row["evidence_end"])]
            ious = [temporal_iou(span, gold) for span in spans]
            conversation_records.append(
                {
                    "question_id": row["question_id"],
                    "transcript_id": tid,
                    "answer": bool(entry.get("answer")),
                    "quotes": kept,
                    "spans": spans,
                    "gold": gold,
                    "candidate_ious": [round(value, 4) for value in ious],
                    "oracle_iou": round(max(ious), 4) if ious else 0.0,
                }
            )
        records.extend(conversation_records)
        print(f"  [{index + 1}/{len(conversations)}] {tid} {elapsed:5.1f}s "
              f"positives={len(conversation_records)} "
              f"candidates={sum(len(r['spans']) for r in conversation_records)}")

    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    (PROJECT_ROOT / "results" / f"{args.tag}_candidates.json").write_text(
        json.dumps(records, indent=2)
    )
    oracle = [r["oracle_iou"] for r in records]
    multi = sum(1 for r in records if len(set(tuple(s) for s in r["spans"])) > 1)
    print(f"positives={len(records)} mean_oracle_iou={statistics.mean(oracle):.3f} "
          f"multi_candidate={multi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
