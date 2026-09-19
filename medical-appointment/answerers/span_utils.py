"""Pure helpers for the ModernBERT answerer: token spans and decoding.

Nothing here imports torch, so the fiddly parts (mapping a gold word range to a
token span inside a passage, mapping a predicted token span back to words, and
the score-aware yes/no rule) are unit-testable without a model.
"""

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

# 3-way classification label order (fixed across training and inference).
LABEL_SUPPORT = 0
LABEL_REFUTE = 1
LABEL_NOT_MENTIONED = 2
LABEL_NAMES = ("support", "refute", "not_mentioned")


def word_char_spans(passage_words: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """Join passage words with single spaces and return each word's char span.

    Building the text and the spans together (rather than searching for words
    afterwards) keeps the mapping exact even where the tokenizer normalises.
    """
    pieces = [word.strip() for word in passage_words]
    text = " ".join(pieces)
    spans: List[Tuple[int, int]] = []
    position = 0
    for piece in pieces:
        spans.append((position, position + len(piece)))
        position += len(piece) + 1
    return text, spans


def _passage_indices(sequence_ids: Sequence[Optional[int]]) -> List[int]:
    return [i for i, seq in enumerate(sequence_ids) if seq == 1]


def best_passage_span(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    sequence_ids: Sequence[Optional[int]],
    max_span_tokens: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Maximize joint start/end score over ordered, non-padding passage tokens."""
    if max_span_tokens is not None and max_span_tokens < 1:
        raise ValueError("max_span_tokens must be positive.")
    starts = deque()
    best = None
    best_score = float("-inf")
    for end in _passage_indices(sequence_ids):
        while starts and start_logits[starts[-1]] < start_logits[end]:
            starts.pop()
        starts.append(end)
        if max_span_tokens is not None:
            while starts and end - starts[0] + 1 > max_span_tokens:
                starts.popleft()
        start = starts[0]
        score = start_logits[start] + end_logits[end]
        if score > best_score:
            best_score = score
            best = (start, end)
    return best


def target_token_span(
    offsets: Sequence[Optional[Tuple[int, int]]],
    sequence_ids: Sequence[Optional[int]],
    char_start: int,
    char_end: int,
) -> Optional[Tuple[int, int]]:
    """Token index range (in the full pair encoding) overlapping a char range.

    Falls back to the passage token whose start is nearest ``char_start`` when
    no token overlaps (e.g. an empty or normalised-away word).
    """
    passage = _passage_indices(sequence_ids)
    if not passage:
        return None

    hits = [
        i for i in passage
        if offsets[i] is not None
        and offsets[i][0] < char_end
        and offsets[i][1] > char_start
    ]
    if hits:
        return hits[0], hits[-1]

    def distance(index: int) -> float:
        offset = offsets[index]
        return abs(offset[0] - char_start) if offset is not None else float("inf")

    best = min(passage, key=distance)
    return best, best


def token_span_to_char(
    offsets: Sequence[Optional[Tuple[int, int]]],
    sequence_ids: Sequence[Optional[int]],
    start_index: int,
    end_index: int,
) -> Optional[Tuple[float, float]]:
    """Char span covered by the predicted start/end tokens (passage side only).

    Returns ``None`` when either index is not a usable passage token.
    """
    passage = set(_passage_indices(sequence_ids))
    if start_index not in passage or end_index not in passage:
        return None
    if start_index > end_index:
        start_index, end_index = end_index, start_index

    start_offset = offsets[start_index]
    end_offset = offsets[end_index]
    if start_offset is None or end_offset is None:
        return None
    return float(start_offset[0]), float(end_offset[1])


def char_span_to_words(
    word_spans: Sequence[Tuple[int, int]], char_start: float, char_end: float
) -> Optional[Tuple[int, int]]:
    """Passage word indices overlapping a char span, or ``None``."""
    hits = [
        i for i, (start, end) in enumerate(word_spans)
        if end > char_start and start < char_end
    ]
    if not hits:
        return None
    return hits[0], hits[-1]


def score_aware_accept(p_support: float, expected_tiou: float) -> bool:
    """The score-aware decision rule: ``yes iff p > 0.4 / (0.8 + 1.2 q)``.

    A missed positive loses accuracy *and* its tIoU, so the cutoff sits below
    0.5; the better the expected span (larger ``q``), the lower it goes.
    ``q`` is clamped to ``[0, 1]``.
    """
    q = min(1.0, max(0.0, expected_tiou))
    return p_support > 0.4 / (0.8 + 1.2 * q)


def psupport_accept(p_support: float, tau: float) -> bool:
    """Plain calibrated-probability rule: ``yes iff p_support > tau``.

    Empirically beat the score-aware rule on the first ModernBERT OOF run
    (LOCO 0.602 vs 0.543): raw ``p_support`` is not yet temperature-calibrated,
    so the q-dependent cutoff was too strict. ``tau`` is chosen out of fold.
    """
    return p_support > tau


def pick_supported_candidate(candidates: Sequence[Dict]) -> Optional[int]:
    """Index of the SUPPORT candidate with the highest predicted tIoU.

    ``candidates`` are dicts with ``label``, ``p_support`` and ``expected_tiou``.
    Returns ``None`` when none is classified SUPPORT.
    """
    support = [
        (c.get("expected_tiou", 0.0), i)
        for i, c in enumerate(candidates)
        if c.get("label") == LABEL_SUPPORT
    ]
    if not support:
        return None
    return max(support)[1]
