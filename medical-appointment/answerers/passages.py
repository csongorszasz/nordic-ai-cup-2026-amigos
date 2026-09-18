"""Sliding-window candidate passages over the ASR word stream.

This is the ModernBERT path's candidate unit, deliberately separate from the
frozen legacy ``windows.py`` (which stays as-is for the served NLI pipeline).

A passage is a contiguous run of ``WINDOW_WORDS`` words advanced by
``STRIDE_WORDS`` (50 % overlap by default). Because every candidate passage is
a contiguous word range, its ``[start, end]`` span maps straight back to the ASR
word timestamps, so a predicted token span inside a passage can be turned into
seconds without a second alignment step.

The caller is responsible for choosing the window/stride: the recall gate in
``answerers.modernbert_data`` measures whether any passage fully contains each
annotated evidence span.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from windows import join_words

# 64/32 (50 % overlap) makes every span up to WINDOW-STRIDE+1 = 33 words
# always fully contained, independent of where it sits on the stride grid. The
# recall gate (T031) showed the originally planned 28/14 left 5/195 gold spans
# unserveable by construction (max gold span 32 words / 14.2 s); 48/24 only
# reached 100 % on this sample by luck of alignment, 64/32 does so structurally.
WINDOW_WORDS = int(os.environ.get("MEDAPP_MB_WINDOW_WORDS", "64"))
STRIDE_WORDS = int(os.environ.get("MEDAPP_MB_STRIDE_WORDS", "32"))


@dataclass
class Passage:
    index: int
    first_word: int
    last_word: int  # inclusive
    start: float
    end: float
    text: str

    def span(self) -> Tuple[float, float]:
        return (self.start, self.end)

    def word_range(self) -> Tuple[int, int]:
        return (self.first_word, self.last_word)


def build_passages(
    words: List[Dict],
    window: int = WINDOW_WORDS,
    stride: int = STRIDE_WORDS,
) -> List[Passage]:
    """Sliding passages with a tail-anchored final window."""
    if not words:
        return []

    window = max(1, window)
    stride = max(1, stride)
    n = len(words)

    ranges: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        last = min(n - 1, i + window - 1)
        ranges.append((i, last))
        if i + window >= n:
            break
        i += stride

    # Anchor the tail so a trailing span is never cut, then drop duplicates.
    tail = (max(0, n - window), n - 1)
    if ranges and ranges[-1] != tail:
        ranges.append(tail)

    seen = set()
    unique: List[Tuple[int, int]] = []
    for pair in ranges:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)

    return [
        Passage(
            index=index,
            first_word=first,
            last_word=last,
            start=float(words[first]["start"]),
            end=float(words[last]["end"]),
            text=join_words(words, first, last),
        )
        for index, (first, last) in enumerate(unique)
    ]


def overlap_word_range(
    words: List[Dict], start: float, end: float
) -> Optional[Tuple[int, int]]:
    """Index range of words overlapping ``[start, end]``, or ``None``."""
    indices = [
        i for i, word in enumerate(words)
        if word["end"] > start and word["start"] < end
    ]
    if not indices:
        return None
    return indices[0], indices[-1]


def contains(passage: Passage, word_range: Tuple[int, int]) -> bool:
    """Whether the passage fully covers a contiguous word range."""
    first, last = word_range
    return passage.first_word <= first and passage.last_word >= last
