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


# --------------------------------------------------------------------------- #
# Sentence-level units (finer index for extraction / reranking)
# --------------------------------------------------------------------------- #

SENTENCE_END = (".", "?", "!")
SENTENCE_PAUSE = 0.6
SENTENCE_MIN_WORDS = 4
SENTENCE_MAX_WORDS = 40


def _sentence_atoms(words: List[Dict]) -> List[List[int]]:
    atoms: List[List[int]] = []
    current: List[int] = []
    for i, word in enumerate(words):
        current.append(i)
        if i + 1 >= len(words):
            break
        nxt = words[i + 1]
        pause = nxt["start"] - word["end"]
        boundary = nxt.get("seg_idx", 0) != word.get("seg_idx", 0)
        ends = word["word"].strip().endswith(SENTENCE_END)
        if (ends and len(current) >= 3) or pause > SENTENCE_PAUSE or boundary:
            atoms.append(current)
            current = []
    if current:
        atoms.append(current)
    return [a for a in atoms if a]


def _merge_short_sentences(words, atoms: List[List[int]]) -> List[List[int]]:
    changed = True
    while changed and len(atoms) > 1:
        changed = False
        for i, atom in enumerate(atoms):
            if len(atom) < SENTENCE_MIN_WORDS:
                if i > 0:
                    atoms[i - 1] = atoms[i - 1] + atom
                    del atoms[i]
                    changed = True
                    break
                if i + 1 < len(atoms):
                    atoms[i] = atom + atoms[i + 1]
                    del atoms[i + 1]
                    changed = True
                    break
    return atoms


def _split_long_sentence(words, atom: List[int]) -> List[List[int]]:
    mid = (len(atom) - 1) / 2
    best_k, best_score = 0, float("-inf")
    for k in range(len(atom) - 1):
        gap = words[atom[k + 1]]["start"] - words[atom[k]]["end"]
        score = gap - 0.02 * abs(k - mid)
        if score > best_score:
            best_score, best_k = score, k
    return [atom[: best_k + 1], atom[best_k + 1:]]


def build_sentences(words: List[Dict]) -> List[Passage]:
    """Sentence-ish units: split on punctuation/pauses, merge short, split long."""
    if not words:
        return []

    atoms = _merge_short_sentences(words, _sentence_atoms(words))
    queue = list(atoms)
    ranges: List[Tuple[int, int]] = []
    while queue:
        atom = queue.pop(0)
        if len(atom) > SENTENCE_MAX_WORDS and len(atom) > 1:
            left, right = _split_long_sentence(words, atom)
            queue.insert(0, left)
            queue.insert(1, right)
        else:
            ranges.append((atom[0], atom[-1]))

    return [
        Passage(
            index=index,
            first_word=first,
            last_word=last,
            start=float(words[first]["start"]),
            end=float(words[last]["end"]),
            text=join_words(words, first, last),
        )
        for index, (first, last) in enumerate(ranges)
    ]


def build_sentence_windows(words: List[Dict], context: int = 1) -> List[Passage]:
    """Overlapping windows of ``context`` sentences each side (containment 0.985 at 1)."""
    sentences = build_sentences(words)
    if not sentences:
        return []

    ranges: List[Tuple[int, int]] = []
    for index in range(len(sentences)):
        lo = max(0, index - context)
        hi = min(len(sentences) - 1, index + context)
        ranges.append((sentences[lo].first_word, sentences[hi].last_word))

    seen = set()
    unique = []
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
