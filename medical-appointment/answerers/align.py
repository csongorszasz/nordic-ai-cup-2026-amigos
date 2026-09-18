"""Align a verbatim quote back to the ASR word stream.

The LLM is asked to copy a contiguous span from the transcript verbatim; this
module maps that string back to ``(start, end)`` seconds and the word index
range. It is deliberately tolerant of whitespace/case and punctuation spacing
(ASR word tokens carry leading spaces).

Two refinements on top of the exact match:

* **Occurrence choice.** A short quote ("twice a day") often occurs more than
  once. When the model also names the transcript segment it copied from, the
  match inside (or nearest to) that segment wins instead of the first one.
* **Fuzzy fallback** (opt-in via ``fuzzy_ratio``). A near-verbatim quote (a
  dropped filler word, "100 mg" vs "100mg") is matched to the most similar word
  range when the exact search fails. A real paraphrase still returns ``None``.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

# A span of audio in seconds from the start of the conversation.
Span = Tuple[float, float]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _segment_distance(words: Sequence[Dict], first: int, last: int, segment: int) -> int:
    """0 when the word range touches ``segment``, else how many segments away."""
    indices = [words[i].get("seg_idx") for i in (first, last)]
    if any(index is None for index in indices):
        return 0
    lo, hi = min(indices), max(indices)
    if lo <= segment <= hi:
        return 0
    return min(abs(segment - lo), abs(segment - hi))


def align_quote(
    words: Sequence[Dict], quote: str, segment: Optional[int] = None
) -> Optional[Tuple[float, float, int, int]]:
    """Verbatim quote -> ``(start, end, first_word, last_word)`` or ``None``.

    With ``segment`` (the transcript line the model says it quoted), the match
    closest to that segment is returned; otherwise the first match.
    """
    if not quote:
        return None

    text = "".join(word["word"] for word in words)
    char_to_word: List[int] = []
    for index, word in enumerate(words):
        char_to_word.extend([index] * len(word["word"]))

    tokens = [_normalise(token) for token in quote.split()]
    tokens = [re.escape(token) for token in tokens if token]
    if not tokens:
        return None

    pattern = re.compile(r"\s*".join(tokens), re.IGNORECASE)
    matches = [
        (char_to_word[match.start()], char_to_word[match.end() - 1])
        for match in pattern.finditer(text)
        if match.end() > match.start()
    ]
    if not matches:
        return None

    if segment is not None and len(matches) > 1:
        # min() is stable, so ties keep the earliest occurrence.
        first, last = min(
            matches, key=lambda pair: _segment_distance(words, pair[0], pair[1], segment)
        )
    else:
        first, last = matches[0]
    return float(words[first]["start"]), float(words[last]["end"]), first, last


def fuzzy_align(
    words: Sequence[Dict],
    quote: str,
    min_ratio: float = 0.85,
    segment: Optional[int] = None,
) -> Optional[Tuple[float, float, int, int]]:
    """Best near-verbatim word range for ``quote``, or ``None`` below ``min_ratio``.

    Compares normalised alphanumeric text over word ranges whose length is
    within a few words of the quote's. Ties prefer the range nearest ``segment``.
    """
    quote_tokens = _TOKEN_RE.findall((quote or "").lower())
    if not quote_tokens or not words:
        return None
    target = " ".join(quote_tokens)
    per_word = [" ".join(_TOKEN_RE.findall(word["word"].lower())) for word in words]

    size = max(1, len(quote.split()))
    slack = max(2, size // 4)
    best = None
    for first in range(len(words)):
        for length in range(max(1, size - slack), size + slack + 1):
            last = first + length - 1
            if last >= len(words):
                break
            candidate = " ".join(t for t in per_word[first:last + 1] if t)
            if not candidate:
                continue
            ratio = SequenceMatcher(None, target, candidate, autojunk=False).ratio()
            distance = (
                _segment_distance(words, first, last, segment) if segment is not None else 0
            )
            key = (ratio, -distance, -first)
            if best is None or key > best[0]:
                best = (key, first, last)
    if best is None or best[0][0] < min_ratio:
        return None
    _, first, last = best
    return float(words[first]["start"]), float(words[last]["end"]), first, last


def apply_offsets(
    span: Optional[Span],
    start_offset: float = 0.0,
    end_offset: float = 0.0,
    duration: Optional[float] = None,
) -> Optional[Span]:
    """Shift a span's boundaries (seconds; negative start = earlier), clamped.

    Matches the annotators' boundary convention, fitted on the training spans by
    ``calibrate_spans.py``. Never returns an inverted span.
    """
    if span is None or (start_offset == 0.0 and end_offset == 0.0):
        return span
    start = max(0.0, span[0] + start_offset)
    end = span[1] + end_offset
    if duration is not None and duration > 0:
        end = min(end, duration)
    if end <= start:
        return span
    return round(start, 3), round(end, 3)


def align_span(
    words: Sequence[Dict],
    quote: str,
    segment: Optional[int] = None,
    fuzzy_ratio: Optional[float] = None,
) -> Optional[Span]:
    """Quote -> ``(start, end)`` seconds, or ``None``.

    Exact match first; with ``fuzzy_ratio`` set, a near-verbatim fallback.
    """
    aligned = align_quote(words, quote, segment=segment)
    if aligned is None and fuzzy_ratio is not None:
        aligned = fuzzy_align(words, quote, min_ratio=fuzzy_ratio, segment=segment)
    if aligned is None:
        return None
    return aligned[0], aligned[1]


def text_between(words: Sequence[Dict], start: float, end: float) -> str:
    """Transcript text overlapping ``[start, end]``."""
    tokens = [
        word["word"] for word in words
        if word["end"] > start and word["start"] < end
    ]
    return _normalise("".join(tokens))


def segment_at(words: Sequence[Dict], start: float, end: float) -> Optional[int]:
    """Segment index holding most of the words in ``[start, end]``."""
    counts: Dict[int, int] = {}
    for word in words:
        if word["end"] > start and word["start"] < end and word.get("seg_idx") is not None:
            counts[word["seg_idx"]] = counts.get(word["seg_idx"], 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda index: counts[index])
