"""Retrieval index and oracle diagnostics for the grounded RAG reader.

Two granularities:

* ``passages``  — the 64/32 sliding windows (gold containment ~0.995-1.000);
* ``sentences`` — sentence +/-``context`` windows (0.985 at context 1).

The oracle measures the best tIoU reachable by any word sub-range inside the
top-k retrieved units, which bounds what *any* grounded reader restricted to
those units can achieve.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .minilm import MiniLMRetriever
from .passages import (
    Passage,
    build_passages,
    build_sentence_windows,
    contains,
    overlap_word_range,
)

INDEX_KINDS = ("passages", "sentences")


def build_index(
    words: List[Dict], kind: str = "passages", context: int = 1
) -> List[Passage]:
    if kind == "sentences":
        return build_sentence_windows(words, context=context)
    if kind == "passages":
        return build_passages(words)
    raise ValueError(f"unknown index kind {kind!r}")


def retrieve(
    question: str,
    index: Sequence[Passage],
    retriever: MiniLMRetriever,
    passage_embeddings=None,
    top_k: Optional[int] = 5,
) -> List[Tuple[Passage, float]]:
    return retriever.rank(
        question, index, top_k=top_k, passage_embeddings=passage_embeddings
    )


def union_oracle_tiou(words: List[Dict], candidates: Sequence[Passage], gold) -> float:
    """Best tIoU of any word sub-range inside the union of the candidates."""
    from dev_eval import _best_subrange

    indices: List[int] = []
    for passage in candidates:
        indices.extend(range(passage.first_word, passage.last_word + 1))
    return _best_subrange(words, sorted(set(indices)), gold)


def gold_rank(
    index: Sequence[Passage],
    words: List[Dict],
    gold: Tuple[float, float],
    ranked: Sequence[Tuple[Passage, float]],
) -> Optional[int]:
    """1-based rank of the first retrieved unit that contains the gold span."""
    word_range = overlap_word_range(words, gold[0], gold[1])
    if word_range is None:
        return None
    for rank, (passage, _) in enumerate(ranked, start=1):
        if contains(passage, word_range):
            return rank
    return None


def oracle_report(
    k_values: Sequence[int] = (1, 3, 5, 8),
    kind: str = "passages",
    context: int = 1,
    limit: Optional[int] = None,
) -> Dict:
    """Selection + union oracle over the annotated positives."""
    from .modernbert_data import load_rows, load_words

    rows = [r for r in load_rows() if r["question_type"] == "positive"]
    if limit:
        rows = rows[:limit]

    retriever = MiniLMRetriever()
    max_k = max(k_values)
    by_tid: Dict[str, List[Dict]] = {}
    for row in rows:
        by_tid.setdefault(row["transcript_id"], []).append(row)

    construction = 0
    total = 0
    ranks: List[Optional[int]] = []
    union: Dict[int, List[float]] = {k: [] for k in k_values}

    for tid, tid_rows in by_tid.items():
        words = load_words(tid)
        index = build_index(words, kind=kind, context=context)
        embeddings = retriever.encode([p.text for p in index])
        for row in tid_rows:
            total += 1
            gold = (float(row["evidence_start"]), float(row["evidence_end"]))
            word_range = overlap_word_range(words, *gold)
            if word_range and any(contains(p, word_range) for p in index):
                construction += 1
            ranked = retrieve(
                row["question"], index, retriever,
                passage_embeddings=embeddings, top_k=max_k,
            )
            ranks.append(gold_rank(index, words, gold, ranked))
            for k in k_values:
                candidates = [p for p, _ in ranked[:k]]
                union[k].append(union_oracle_tiou(words, candidates, gold))

    return {
        "kind": kind,
        "context": context,
        "k_values": list(k_values),
        "units_per_conversation": sum(
            len(build_index(load_words(tid), kind=kind, context=context))
            for tid in by_tid
        ) / max(1, len(by_tid)),
        "positives": total,
        "construction": construction,
        "construction_rate": round(construction / total, 4) if total else 0.0,
        "top1_rate": round(sum(1 for r in ranks if r == 1) / total, 4) if total else 0.0,
        "top3_rate": round(sum(1 for r in ranks if r and r <= 3) / total, 4) if total else 0.0,
        "union_oracle": {
            k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in union.items()
        },
    }


def print_report(report: Dict) -> None:
    print(
        f"\n[{report['kind']} context={report['context']}] "
        f"{report['units_per_conversation']:.1f} units/conversation  "
        f"construction {report['construction_rate']:.3f}  "
        f"gold passage top1 {report['top1_rate']:.3f} top3 {report['top3_rate']:.3f}"
    )
    for k, value in report["union_oracle"].items():
        print(f"  top-{k} union oracle tIoU {value:.3f}")
