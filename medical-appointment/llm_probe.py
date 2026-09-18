"""LLM ceiling probe: L0 (zero-shot), L1 (few-shot), L2 (two-pass).

Runs a local instruct model over the supplied conversations and scores it with
the same ``local_evaluator.Statistics`` the rest of the project uses. The score
is in-sample/ceiling (the 39 conversations carry labels), comparable to the
ModernBERT OOF 0.610 and hybrid 0.647-0.661.

    python llm_probe.py --rungs L0 L1 L2 --limit 3        # smoke
    python llm_probe.py --rungs L0 L1 L2                  # full 39

Writes ``results/llm_<tag>_<rung>.json`` (summary) and
``results/llm_<tag>_<rung>_questions.json`` (per-question records).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answerers import modernbert_data as data  # noqa: E402
from answerers.align import align_span  # noqa: E402
from answerers.llm_client import HFClient  # noqa: E402
from answerers.llm_parse import parse_answers, parse_decisions  # noqa: E402
from answerers.llm_prompt import (  # noqa: E402
    build_few_shot,
    build_l0_messages,
    build_l1_messages,
    build_l2_cite_messages,
    build_l2_decide_messages,
    qid_for,
)
from local_evaluator import UNANSWERED, Statistics  # noqa: E402
from utils import gold_evidence  # noqa: E402

RUNGS = ("L0", "L1", "L2")


def load_transcript(transcript_id: str) -> Dict:
    """Full transcript dict (segments + words), preferring the large-v3 cache."""
    preferred = data.TRANSCRIPTS / f"conversation_{transcript_id}.dc5ba020.json"
    path = preferred if preferred.exists() else sorted(
        data.TRANSCRIPTS.glob(f"conversation_{transcript_id}.*.json")
    )[0]
    return json.loads(path.read_text())


def _documents(limit: Optional[int]):
    rows = data.load_rows()
    rows_by_tid: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        rows_by_tid[row["transcript_id"]].append(row)
    conversations = list(rows_by_tid.items())
    if limit:
        conversations = conversations[:limit]
    transcripts = {tid: load_transcript(tid) for tid, _ in conversations}
    return conversations, rows_by_tid, transcripts


def _decide_and_cite(
    rung: str,
    client,
    transcript: Dict,
    rows: List[Dict],
    few_shot: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict], float, int]:
    """Return per-question records, elapsed seconds, parse failures."""
    questions = [row["question"] for row in rows]
    ids = [qid_for(index) for index in range(len(questions))]

    started = time.perf_counter()
    if rung == "L0":
        raw = client.generate(build_l0_messages(transcript, questions))
        parsed = parse_answers(raw, ids)
        entries = {qid: parsed.get(qid) for qid in ids}
    elif rung == "L1":
        raw = client.generate(
            build_l1_messages(transcript, questions, few_shot)
        )
        parsed = parse_answers(raw, ids)
        entries = {qid: parsed.get(qid) for qid in ids}
    else:  # L2 two-pass
        decide_raw = client.generate(
            build_l2_decide_messages(transcript, questions, few_shot)
        )
        decisions = parse_decisions(decide_raw, ids)
        yes = [
            (qid, question)
            for qid, question in zip(ids, questions)
            if decisions.get(qid) is True
        ]
        cites: Dict[str, Optional[dict]] = {}
        if yes:
            cite_raw = client.generate(
                build_l2_cite_messages(transcript, yes)
            )
            cites = parse_answers(cite_raw, [qid for qid, _ in yes])
        entries = {}
        for qid in ids:
            decision = decisions.get(qid)
            if decision is None:
                entries[qid] = None
            elif decision is False:
                entries[qid] = {"answer": False, "quote": None}
            else:
                entries[qid] = {"answer": True, "quote": (cites.get(qid) or {}).get("quote")}
    elapsed = time.perf_counter() - started

    words = transcript.get("words", [])
    records: List[Dict] = []
    parse_failures = 0
    for index, (row, qid) in enumerate(zip(rows, ids)):
        entry = entries.get(qid)
        if entry is None or entry.get("answer") is None:
            parse_failures += 1
            prediction, span, quote = UNANSWERED, None, None
        elif entry["answer"]:
            quote = entry.get("quote")
            span = align_span(words, quote or "")
            prediction = 1
        else:
            prediction, span, quote = 0, None, None
        records.append(
            {
                "question_id": row["question_id"],
                "transcript_id": row["transcript_id"],
                "question_type": row["question_type"],
                "label": int(row["label"]),
                "prediction": int(prediction),
                "answer": bool(prediction == 1),
                "span": list(span) if span is not None else None,
                "gold": (
                    [float(row["evidence_start"]), float(row["evidence_end"])]
                    if row["question_type"] == "positive"
                    else None
                ),
                "quote": quote,
                "rung": rung,
            }
        )
    return records, elapsed, parse_failures


def score_records(records: List[Dict]) -> Dict:
    statistics = Statistics()
    for record in records:
        gold = tuple(record["gold"]) if record["gold"] else None
        span = tuple(record["span"]) if record["span"] else None
        statistics.record(
            record["question_type"], record["label"],
            record["prediction"], gold, span,
        )
    positives = [r for r in records if r["label"] == 1]
    yes_positives = [r for r in positives if r["answer"]]
    yes_answers = [r for r in records if r["answer"]]
    found = [r for r in yes_answers if r["span"] is not None]
    by_type = {
        name: {"correct": value[0], "total": value[1]}
        for name, value in statistics.by_type.items()
    }
    return {
        "score": round(statistics.final_score, 4),
        "accuracy": round(statistics.accuracy, 4),
        "mean_tiou": round(statistics.mean_tiou, 4),
        "by_type": by_type,
        "positive_recall": round(len(yes_positives) / len(positives), 4)
        if positives else 0.0,
        "quote_found_rate": round(len(found) / len(yes_answers), 4)
        if yes_answers else 0.0,
        "yes_answers": len(yes_answers),
        "tious_answered_yes": round(statistics.mean_tiou_answered_yes, 4),
        "questions": len(records),
    }


def run_rung(rung: str, client, limit: Optional[int], tag: str) -> Dict:
    conversations, rows_by_tid, transcripts = _documents(limit)
    evidence = data.load_evidence()

    print(f"\n=== {rung} ({len(conversations)} conversations) ===")
    records: List[Dict] = []
    parse_failures = 0
    latencies: List[float] = []
    for index, (tid, rows) in enumerate(conversations):
        if rung == "L0":
            few_shot: Sequence[Tuple[str, str]] = ()
        else:
            few_shot = build_few_shot(
                rows_by_tid, transcripts, evidence, exclude_tid=tid
            )
        conversation_records, elapsed, failures = _decide_and_cite(
            rung, client, transcripts[tid], rows, few_shot
        )
        records.extend(conversation_records)
        latencies.append(elapsed)
        parse_failures += failures
        print(
            f"  [{index + 1}/{len(conversations)}] {tid} "
            f"{elapsed:5.1f}s yes={sum(r['answer'] for r in conversation_records)}/10 "
            f"parse_fail={failures}"
        )

    summary = score_records(records)
    summary.update(
        {
            "rung": rung,
            "tag": tag,
            "model": getattr(client, "model_name", "stub"),
            "conversations": len(conversations),
            "parse_failures": parse_failures,
            "latency_mean_s": round(sum(latencies) / len(latencies), 2)
            if latencies else 0.0,
            "latency_worst_s": round(max(latencies), 2) if latencies else 0.0,
        }
    )
    print(
        f"  -> score {summary['score']:.3f}  acc {summary['accuracy']:.3f}  "
        f"mIoU {summary['mean_tiou']:.3f}  recall {summary['positive_recall']:.3f}  "
        f"quote_found {summary['quote_found_rate']:.3f}  "
        f"parse_fail {parse_failures}"
    )

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"llm_{tag}_{rung}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"llm_{tag}_{rung}_questions.json").write_text(
        json.dumps(records, indent=2)
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs", nargs="+", choices=RUNGS, default=list(RUNGS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="probe")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    client = HFClient(model_name=args.model, max_new_tokens=args.max_new_tokens)
    client.warm_up()

    summaries = []
    for rung in args.rungs:
        summaries.append(run_rung(rung, client, args.limit, args.tag))

    print("\n=== summary ===")
    print(f"{'rung':<5}{'score':>7}{'acc':>7}{'mIoU':>7}{'recall':>8}{'quote_f':>9}")
    for summary in summaries:
        print(
            f"{summary['rung']:<5}{summary['score']:>7.3f}"
            f"{summary['accuracy']:>7.3f}{summary['mean_tiou']:>7.3f}"
            f"{summary['positive_recall']:>8.3f}{summary['quote_found_rate']:>9.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
