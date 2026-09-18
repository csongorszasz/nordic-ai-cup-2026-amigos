"""Dataset build and the retrieval recall gate for the ModernBERT answerer.

Two jobs live here:

1. **Recall gate** (``recall_gate``): confirm that the sliding passages plus
   MiniLM retrieval actually surface the annotated evidence before a model is
   trained. If the gold span is not inside the top-k passages for a positive,
   no scorer downstream can recover it. This is the gate the plan requires.

2. **Candidate examples** (``build_examples``): per-question candidate passages
   with a 3-way label and a word-level span target, ready for ModernBERT. Labels
   come from ``annotations/evidence.csv`` (support = gold span, refute = agent
   span, absent = off-topic). Cross-conversation passages provide
   NOT_MENTIONED negatives.

Run the gate::

    python -m answerers.modernbert_data --sweep
"""

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import temporal_iou

from .minilm import MiniLMRetriever
from .passages import (
    STRIDE_WORDS,
    WINDOW_WORDS,
    Passage,
    build_passages,
    contains,
    overlap_word_range,
)

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
QUESTIONS = ROOT / "data" / "question_train.csv"
EVIDENCE = ROOT / "annotations" / "evidence.csv"

# Retrieval depth handed to the cross-encoder. 8 keeps the gate near-perfect
# (T031: support 0.995 / refute 1.000); 5 is the latency fallback on the 1650.
TOP_K = int(os.environ.get("MEDAPP_MB_TOP_K", "8"))


def load_words(transcript_id: str) -> List[Dict]:
    """Word list for a conversation, preferring the large-v3 transcript."""
    preferred = TRANSCRIPTS / f"conversation_{transcript_id}.dc5ba020.json"
    path = preferred if preferred.exists() else sorted(
        TRANSCRIPTS.glob(f"conversation_{transcript_id}.*.json")
    )[0]
    return json.loads(path.read_text())["words"]


def load_transcript(transcript_id: str) -> Dict:
    """Full transcript dict (segments + words), preferring the large-v3 cache."""
    preferred = TRANSCRIPTS / f"conversation_{transcript_id}.dc5ba020.json"
    path = preferred if preferred.exists() else sorted(
        TRANSCRIPTS.glob(f"conversation_{transcript_id}.*.json")
    )[0]
    return json.loads(path.read_text())


def load_rows() -> List[Dict[str, str]]:
    with open(QUESTIONS, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_evidence() -> Dict[str, Dict[str, str]]:
    with open(EVIDENCE, newline="", encoding="utf-8") as handle:
        return {row["question_id"]: row for row in csv.DictReader(handle)}


def target_span(row: Dict[str, str], evidence: Dict[str, str]) -> Optional[Tuple[float, float]]:
    """The span a question's evidence points at, or ``None``.

    Positives use the supplied gold span; hard negatives use the agent's refute
    span (from ``evidence.csv``) when one exists; off-topic and absent have none.
    """
    if row["question_type"] == "positive":
        return float(row["evidence_start"]), float(row["evidence_end"])

    if row["question_type"] == "hard_negative":
        item = evidence.get(row["question_id"], {})
        if item.get("bucket") == "refute" and item.get("start") and item.get("end"):
            return float(item["start"]), float(item["end"])
    return None


def bucket_label(row: Dict[str, str], evidence: Dict[str, str]) -> str:
    if row["question_type"] == "positive":
        return "support"
    if row["question_type"] == "hard_negative":
        item = evidence.get(row["question_id"], {})
        return "refute" if item.get("bucket") == "refute" else "not_mentioned"
    return "not_mentioned"


def _load_word_lists(rows: List[Dict[str, str]]) -> Dict[str, List[Dict]]:
    cache: Dict[str, List[Dict]] = {}
    for row in rows:
        tid = row["transcript_id"]
        if tid not in cache:
            cache[tid] = load_words(tid)
    return cache


def recall_gate(
    k_values: Tuple[int, ...] = (1, 3, 5),
    window: int = WINDOW_WORDS,
    stride: int = STRIDE_WORDS,
    limit: Optional[int] = None,
    retriever: Optional[MiniLMRetriever] = None,
) -> Dict:
    """Measure gold/refute containment at construction and after retrieval."""
    rows = load_rows()
    evidence = load_evidence()
    if limit:
        rows = rows[:limit]

    words_by_tid = _load_word_lists(rows)
    passages_by_tid = {
        tid: build_passages(words, window=window, stride=stride)
        for tid, words in words_by_tid.items()
    }
    retriever = retriever or MiniLMRetriever()

    max_k = max(k_values)
    support_total = support_construction = 0
    refute_total = refute_construction = 0
    support_hits = Counter()
    refute_hits = Counter()

    embeddings_cache: Dict[str, object] = {}

    for row in rows:
        target = target_span(row, evidence)
        if target is None:
            continue
        label = bucket_label(row, evidence)
        if label not in ("support", "refute"):
            continue

        tid = row["transcript_id"]
        words = words_by_tid[tid]
        passages = passages_by_tid[tid]
        word_range = overlap_word_range(words, target[0], target[1])
        if word_range is None:
            continue

        construction = any(contains(p, word_range) for p in passages)

        if tid not in embeddings_cache:
            embeddings_cache[tid] = retriever.encode([p.text for p in passages])
        ranked = retriever.rank(
            row["question"],
            passages,
            top_k=max_k,
            passage_embeddings=embeddings_cache[tid],
        )

        if label == "support":
            support_total += 1
            support_construction += int(construction)
            bucket = support_hits
        else:
            refute_total += 1
            refute_construction += int(construction)
            bucket = refute_hits

        for k in k_values:
            if any(contains(p, word_range) for p, _ in ranked[:k]):
                bucket[k] += 1

    report = {
        "window": window,
        "stride": stride,
        "k_values": list(k_values),
        "passages_per_conversation": (
            sum(len(p) for p in passages_by_tid.values()) / max(1, len(passages_by_tid))
        ),
        "support_total": support_total,
        "support_construction": support_construction,
        "support_hits": dict(support_hits),
        "refute_total": refute_total,
        "refute_construction": refute_construction,
        "refute_hits": dict(refute_hits),
    }
    return report


def _format_rate(hit: int, total: int) -> str:
    return f"{hit}/{total} ({hit / total:.3f})" if total else f"{hit}/{total} (n/a)"


def print_report(report: Dict) -> None:
    print(
        f"\nRecall gate  window={report['window']} stride={report['stride']}  "
        f"passages/conversation {report['passages_per_conversation']:.1f}"
    )
    print(
        f"  supports  construction {_format_rate(report['support_construction'], report['support_total'])}"
    )
    for k in report["k_values"]:
        print(
            f"            top-{k} retrieval {_format_rate(report['support_hits'].get(k, 0), report['support_total'])}"
        )
    if report["refute_total"]:
        print(
            f"  refutes   construction {_format_rate(report['refute_construction'], report['refute_total'])}"
        )
        for k in report["k_values"]:
            print(
                f"            top-{k} retrieval {_format_rate(report['refute_hits'].get(k, 0), report['refute_total'])}"
            )


def build_examples(
    window: int = WINDOW_WORDS,
    stride: int = STRIDE_WORDS,
    top_k: int = TOP_K,
    n_cross_negatives: int = 2,
    limit: Optional[int] = None,
    retriever: Optional[MiniLMRetriever] = None,
    seed: int = 13,
) -> List[Dict]:
    """Per-question candidate passages with labels and span targets.

    Every candidate carries:

    * ``label``: ``support`` / ``refute`` / ``not_mentioned``;
    * ``span_words``: the target word range *relative to the passage*, or None;
    * ``span``: the target ``[start, end]`` in seconds, or None;
    * ``target_tiou``: tIoU between the passage span and the target span (0 for
      not-mentioned negatives).

    Retrieval supplies the in-conversation candidates; cross-conversation
    passages supply additional NOT_MENTIONED negatives.
    """
    rng = random.Random(seed)
    rows = load_rows()
    evidence = load_evidence()
    if limit:
        rows = rows[:limit]

    words_by_tid = _load_word_lists(rows)
    passages_by_tid = {
        tid: build_passages(words, window=window, stride=stride)
        for tid, words in words_by_tid.items()
    }
    retriever = retriever or MiniLMRetriever()
    tids = sorted(passages_by_tid)

    embeddings_cache: Dict[str, object] = {}
    examples: List[Dict] = []

    for row in rows:
        tid = row["transcript_id"]
        words = words_by_tid[tid]
        passages = passages_by_tid[tid]
        target = target_span(row, evidence)
        target_range = (
            overlap_word_range(words, target[0], target[1]) if target else None
        )
        wanted = bucket_label(row, evidence)

        if tid not in embeddings_cache:
            embeddings_cache[tid] = retriever.encode([p.text for p in passages])
        ranked = retriever.rank(
            row["question"],
            passages,
            top_k=top_k,
            passage_embeddings=embeddings_cache[tid],
        )

        for passage, score in ranked:
            hit = (
                target_range is not None
                and wanted in ("support", "refute")
                and contains(passage, target_range)
            )
            label = wanted if hit else "not_mentioned"
            span_words = None
            span = None
            target_tiou = 0.0
            if hit:
                first, last = target_range
                span_words = [first - passage.first_word, last - passage.first_word]
                span = [target[0], target[1]]
                target_tiou = temporal_iou(passage.span(), target)
            examples.append(
                {
                    "question_id": row["question_id"],
                    "transcript_id": tid,
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "passage_index": passage.index,
                    "passage": passage.text,
                    "passage_words": [
                        words[k]["word"].strip()
                        for k in range(passage.first_word, passage.last_word + 1)
                    ],
                    "passage_span": list(passage.span()),
                    "retrieval_score": round(score, 4),
                    "label": label,
                    "span_words": span_words,
                    "span": span,
                    "target_tiou": round(target_tiou, 4),
                }
            )

        # Cross-conversation negatives: definitely not mentioned.
        other_tids = [t for t in tids if t != tid]
        for _ in range(n_cross_negatives):
            if not other_tids:
                break
            other = rng.choice(other_tids)
            passage = rng.choice(passages_by_tid[other])
            examples.append(
                {
                    "question_id": row["question_id"],
                    "transcript_id": other,
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "passage_index": passage.index,
                    "passage": passage.text,
                    "passage_words": [
                        words_by_tid[other][k]["word"].strip()
                        for k in range(passage.first_word, passage.last_word + 1)
                    ],
                    "passage_span": list(passage.span()),
                    "retrieval_score": None,
                    "label": "not_mentioned",
                    "span_words": None,
                    "span": None,
                    "target_tiou": 0.0,
                }
            )

    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=WINDOW_WORDS)
    parser.add_argument("--stride", type=int, default=STRIDE_WORDS)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sweep", action="store_true",
        help="Compare the spec window (28/14) with full-construction sizes.",
    )
    parser.add_argument(
        "--dump-examples", default=None,
        help="Write candidate examples to this JSON path instead of the gate.",
    )
    args = parser.parse_args()

    if args.dump_examples:
        examples = build_examples(limit=args.limit)
        Path(args.dump_examples).write_text(json.dumps(examples, indent=2))
        counts = Counter(e["label"] for e in examples)
        print(f"wrote {len(examples)} examples -> {args.dump_examples}")
        print("labels:", dict(counts))
        return 0

    retriever = MiniLMRetriever()
    if args.sweep:
        for window, stride in ((28, 14), (36, 18), (48, 24)):
            report = recall_gate(
                tuple(args.k), window=window, stride=stride,
                limit=args.limit, retriever=retriever,
            )
            print_report(report)
    else:
        report = recall_gate(
            tuple(args.k), window=args.window, stride=args.stride,
            limit=args.limit, retriever=retriever,
        )
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
