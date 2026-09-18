"""Align a verbatim quote back to the ASR word stream.

The LLM is asked to copy a contiguous span from the transcript verbatim; this
module maps that string back to ``(start, end)`` seconds and the word index
range. It is deliberately tolerant of whitespace/case and punctuation spacing
(ASR word tokens carry leading spaces), but it will not match a paraphrase — a
quote that is not found returns ``None``, which is the honest outcome.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

# A span of audio in seconds from the start of the conversation.
Span = Tuple[float, float]


def _normalise(text: str) -> str:
    return " ".join(text.split())


def align_quote(
    words: Sequence[Dict], quote: str
) -> Optional[Tuple[float, float, int, int]]:
    """Verbatim quote -> ``(start, end, first_word, last_word)`` or ``None``."""
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

    pattern = r"\s*".join(tokens)
    match = re.compile(pattern, re.IGNORECASE).search(text)
    if not match:
        return None

    first = char_to_word[match.start()]
    last = char_to_word[match.end() - 1]
    return float(words[first]["start"]), float(words[last]["end"]), first, last


def align_span(words: Sequence[Dict], quote: str) -> Optional[Span]:
    """Verbatim quote -> ``(start, end)`` seconds, or ``None``."""
    aligned = align_quote(words, quote)
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
