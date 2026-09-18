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
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answerers import modernbert_data as data  # noqa: E402
from answerers import rag as rag_module  # noqa: E402
from answerers.align import align_span, text_between  # noqa: E402
from answerers.llm_client import HFClient  # noqa: E402
from answerers.llm_parse import parse_answers, parse_decisions  # noqa: E402
from answerers.llm_prompt import (  # noqa: E402
    build_few_shot,
    build_l0_messages,
    build_l1_messages,
    build_l2_cite_messages,
    build_l2_decide_messages,
    build_rag_messages,
    qid_for,
    render_rag_example,
)
from answerers.minilm import MiniLMRetriever  # noqa: E402
from answerers.passages import contains, overlap_word_range  # noqa: E402
from local_evaluator import UNANSWERED, Statistics  # noqa: E402
from utils import gold_evidence  # noqa: E402

RUNGS = ("L0", "L1", "L2", "RAG")

logger = logging.getLogger(__name__)


def _generate(client, messages) -> str:
    """Generation that never aborts the run (an OOM must not cost every question)."""
    try:
        return client.generate(messages)
    except Exception:
        logger.exception("generation failed")
        return ""


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
    # Load every transcript so few-shot examples can be drawn from any OTHER
    # conversation; only the first `limit` are scored.
    transcripts = {tid: load_transcript(tid) for tid in rows_by_tid}
    conversations = list(rows_by_tid.items())
    if limit:
        conversations = conversations[:limit]
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
        raw = _generate(client, build_l0_messages(transcript, questions))
        parsed = parse_answers(raw, ids)
        entries = {qid: parsed.get(qid) for qid in ids}
    elif rung == "L1":
        raw = _generate(client, build_l1_messages(transcript, questions, few_shot))
        parsed = parse_answers(raw, ids)
        entries = {qid: parsed.get(qid) for qid in ids}
    else:  # L2 two-pass
        decide_raw = _generate(client, build_l2_decide_messages(transcript, questions, few_shot))
        decisions = parse_decisions(decide_raw, ids)
        yes = [
            (qid, question)
            for qid, question in zip(ids, questions)
            if decisions.get(qid) is True
        ]
        cites: Dict[str, Optional[dict]] = {}
        if yes:
            cite_raw = _generate(client, build_l2_cite_messages(transcript, yes))
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


def _norm(text: Optional[str]) -> str:
    return " ".join((text or "").split()).lower()


def _candidate_id_for(candidates, words, span):
    word_range = overlap_word_range(words, span[0], span[1])
    if word_range is None:
        return None
    for index, passage in enumerate(candidates):
        if contains(passage, word_range):
            return f"c{index + 1:02d}"
    return None


def _rag_few_shot(
    rows_by_tid, transcripts, evidence, exclude_tid, retriever, top_k, cache
):
    """Balanced LOCO-safe candidate-citing examples for the RAG rung."""
    examples: List[Tuple[str, str]] = []
    used = {exclude_tid}

    def index_for(tid):
        if tid not in cache:
            words = transcripts[tid].get("words", [])
            built = rag_module.build_index(words, kind="passages", context=1)
            cache[tid] = (words, built, retriever.encode([p.text for p in built]))
        return cache[tid]

    def pick(question_type, require_refute=False):
        for tid in sorted(rows_by_tid):
            if tid in used:
                continue
            for row in rows_by_tid[tid]:
                if row["question_type"] != question_type:
                    continue
                if require_refute and evidence.get(
                    row["question_id"], {}
                ).get("bucket") != "refute":
                    continue
                return row
        return None

    def add(row, question_type):
        if row is None or row["transcript_id"] not in transcripts:
            return
        tid = row["transcript_id"]
        used.add(tid)
        words, built, embeddings = index_for(tid)
        candidates = [
            p for p, _ in rag_module.retrieve(
                row["question"], built, retriever,
                passage_embeddings=embeddings, top_k=top_k,
            )
        ]
        if not candidates:
            return
        if question_type == "positive":
            span = (float(row["evidence_start"]), float(row["evidence_end"]))
            candidate_id = _candidate_id_for(candidates, words, span) or "c01"
            examples.append(render_rag_example(
                row["question"], candidates, True, candidate_id,
                text_between(words, span[0], span[1]),
            ))
        else:
            examples.append(
                render_rag_example(row["question"], candidates, False, None, None)
            )

    add(pick("positive"), "positive")
    add(pick("hard_negative", require_refute=True), "hard_negative")
    add(pick("off_topic"), "off_topic")
    return examples


def _run_rag(client, transcript, rows, index, retriever, top_k, few_shot=()):
    """Grounded RAG reader: answer + candidate id + verbatim quote."""
    words = transcript.get("words", [])
    questions = [row["question"] for row in rows]
    ids = [qid_for(i) for i in range(len(questions))]

    embeddings = retriever.encode([p.text for p in index])
    candidates_by_index = []
    for question in questions:
        ranked = rag_module.retrieve(
            question, index, retriever, passage_embeddings=embeddings, top_k=top_k
        )
        candidates_by_index.append([p for p, _ in ranked])

    started = time.perf_counter()
    raw = _generate(client, build_rag_messages(questions, candidates_by_index, few_shot))
    elapsed = time.perf_counter() - started
    parsed = parse_answers(raw, ids)

    records: List[Dict] = []
    parse_failures = 0
    for i, (row, qid) in enumerate(zip(rows, ids)):
        entry = parsed.get(qid)
        cands = candidates_by_index[i]
        gold = (
            (float(row["evidence_start"]), float(row["evidence_end"]))
            if row["question_type"] == "positive" else None
        )
        gold_range = overlap_word_range(words, *gold) if gold else None

        if entry is None or entry.get("answer") is None:
            parse_failures += 1
            fields = dict(prediction=UNANSWERED, span=None, quote=None,
                          candidate=None, grounded=False, cited_contains_gold=False)
        elif entry["answer"]:
            quote = entry.get("quote")
            cand_id = entry.get("candidate")
            digits = "".join(ch for ch in str(cand_id or "") if ch.isdigit())
            cidx = int(digits) - 1 if digits else None
            passage = (
                cands[cidx] if cidx is not None and 0 <= cidx < len(cands) else None
            )
            span = align_span(words, quote or "")
            grounded = bool(passage and quote and _norm(quote) in _norm(passage.text))
            cited_gold = bool(
                passage and gold_range and contains(passage, gold_range)
            )
            fields = dict(prediction=1, span=span, quote=quote, candidate=cand_id,
                          grounded=grounded, cited_contains_gold=cited_gold)
        else:
            fields = dict(prediction=0, span=None, quote=None, candidate=None,
                          grounded=True, cited_contains_gold=False)

        record = {
            "question_id": row["question_id"],
            "transcript_id": row["transcript_id"],
            "question_type": row["question_type"],
            "label": int(row["label"]),
            "answer": bool(fields["prediction"] == 1),
            "span": list(fields["span"]) if fields["span"] is not None else None,
            "gold": (
                [float(row["evidence_start"]), float(row["evidence_end"])]
                if row["question_type"] == "positive" else None
            ),
            "rung": "RAG",
        }
        record.update(fields)
        records.append(record)
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
    summary = {
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
    cited = [r for r in yes_answers if r.get("candidate")]
    if cited:
        summary["passage_selection_accuracy"] = round(
            sum(1 for r in cited if r.get("cited_contains_gold")) / len(cited), 4
        )
        summary["grounded_rate"] = round(
            sum(1 for r in cited if r.get("grounded")) / len(cited), 4
        )
    return summary


def run_rung(
    rung: str,
    client,
    limit: Optional[int],
    tag: str,
    index_kind: str = "passages",
    top_k: int = 5,
    context: int = 1,
) -> Dict:
    conversations, rows_by_tid, transcripts = _documents(limit)
    evidence = data.load_evidence()
    retriever = MiniLMRetriever() if rung == "RAG" else None
    index_cache: Dict[str, list] = {}
    example_cache: Dict[str, list] = {}

    print(f"\n=== {rung} ({len(conversations)} conversations) ===")
    records: List[Dict] = []
    parse_failures = 0
    latencies: List[float] = []
    for conversation_index, (tid, rows) in enumerate(conversations):
        if rung == "RAG":
            words = transcripts[tid].get("words", [])
            if tid not in index_cache:
                index_cache[tid] = rag_module.build_index(
                    words, kind=index_kind, context=context
                )
            few_shot = _rag_few_shot(
                rows_by_tid, transcripts, evidence, tid, retriever, top_k,
                example_cache,
            )
            conversation_records, elapsed, failures = _run_rag(
                client, transcripts[tid], rows, index_cache[tid], retriever,
                top_k, few_shot,
            )
        else:
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
            f"  [{conversation_index + 1}/{len(conversations)}] {tid} "
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
    parser.add_argument("--index", default="passages", choices=["passages", "sentences"],
                        help="RAG candidate granularity.")
    parser.add_argument("--top-k", type=int, default=5, help="RAG candidates per question.")
    parser.add_argument("--context", type=int, default=1,
                        help="Sentence window half-width for --index sentences.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    client = HFClient(model_name=args.model, max_new_tokens=args.max_new_tokens)
    client.warm_up()

    summaries = []
    for rung in args.rungs:
        summaries.append(run_rung(
            rung, client, args.limit, args.tag,
            index_kind=args.index, top_k=args.top_k, context=args.context,
        ))

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
