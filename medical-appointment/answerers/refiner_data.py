"""Anchored-window examples for the post-hoc boundary refiner.

For every positive the E4B probe answered yes, the window is centred on the
LLM quote's aligned word range and the target is the annotated gold word range
inside that window. The refiner learns only *how much* of the cited passage a
human marks — not which occurrence to cite; positives whose gold lies outside
the LLM's window are dropped and counted in the build report.

No retrieval or MiniLM is used, and nothing here imports torch.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from utils import temporal_iou

from .align import align_quote
from .modernbert_data import load_rows, load_words
from .passages import overlap_word_range
from .refiner import window_bounds

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LLM_QUESTIONS = str(ROOT / "results" / "llm_gemma_l1_L1_questions.json")


def load_llm_records(path: str) -> Dict[str, Dict]:
    """Question-id -> probe record from an ``llm_*_questions.json`` file."""
    records = json.loads(Path(path).read_text())
    return {record["question_id"]: record for record in records}


def build_refiner_examples(
    llm_questions_path: str = DEFAULT_LLM_QUESTIONS,
    window: int = 24,
    limit: Optional[int] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Anchored examples with a relative gold span and the build report.

    Every example carries ``passage_words`` (the window) and ``span_words``
    (gold relative to the window), so the existing ``WordSpanDataset`` and
    ``Collator`` can consume it unchanged.
    """
    rows = load_rows()
    if limit:
        rows = rows[:limit]
    records = load_llm_records(llm_questions_path)

    examples: List[Dict] = []
    skipped: Counter = Counter()
    for row in rows:
        if row["question_type"] != "positive":
            continue
        record = records.get(row["question_id"])
        if (
            record is None
            or not record.get("answer")
            or not record.get("quote")
            or not record.get("span")
        ):
            skipped["llm_not_answered_yes"] += 1
            continue

        tid = row["transcript_id"]
        words = load_words(tid)
        aligned = align_quote(words, record["quote"])
        if aligned is None:
            skipped["quote_unaligned"] += 1
            continue
        _, _, llm_first, llm_last = aligned

        gold_span = (float(row["evidence_start"]), float(row["evidence_end"]))
        gold_range = overlap_word_range(words, gold_span[0], gold_span[1])
        if gold_range is None:
            skipped["gold_without_words"] += 1
            continue

        lo, hi = window_bounds(len(words), llm_first, llm_last, window)
        first, last = gold_range
        if last < lo or first > hi:
            skipped["gold_outside_window"] += 1
            continue

        relative_first = max(0, first - lo)
        relative_last = min(hi, last) - lo
        window_span = (float(words[lo]["start"]), float(words[hi]["end"]))
        llm_span = (
            float(words[llm_first]["start"]),
            float(words[llm_last]["end"]),
        )
        examples.append(
            {
                "question_id": row["question_id"],
                "transcript_id": tid,
                "question": row["question"],
                "question_type": "positive",
                "first_word": lo,
                "last_word": hi,
                "passage_words": [
                    words[k]["word"].strip() for k in range(lo, hi + 1)
                ],
                "window_span": list(window_span),
                "span_words": [relative_first, relative_last],
                "span": list(gold_span),
                "gold_word_range": [first, last],
                "llm_word_range": [llm_first, llm_last],
                "llm_span": list(llm_span),
                "target_tiou": round(temporal_iou(window_span, gold_span), 4),
                "label": "support",
            }
        )
    return examples, dict(skipped)
