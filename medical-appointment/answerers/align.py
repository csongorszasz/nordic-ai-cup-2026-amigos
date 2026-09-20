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


def align_quote_matches(
    words: Sequence[Dict],
    quote: str,
    *,
    first_word: int = 0,
    last_word: Optional[int] = None,
) -> List[Tuple[float, float, int, int]]:
    """All whole-word quote matches inside an optional word range."""
    if not quote:
        return []
    if not words:
        return []

    last_word = len(words) - 1 if last_word is None else last_word
    first_word = max(0, first_word)
    last_word = min(len(words) - 1, last_word)
    if first_word > last_word:
        return []

    text = "".join(word["word"] for word in words)
    char_to_word: List[int] = []
    word_bounds: List[Tuple[int, int]] = []
    position = 0
    for index, word in enumerate(words):
        token = word["word"]
        char_to_word.extend([index] * len(token))
        left = position + len(token) - len(token.lstrip())
        right = position + len(token.rstrip())
        word_bounds.append((left, right))
        position += len(token)

    tokens = [_normalise(token) for token in quote.split()]
    tokens = [re.escape(token) for token in tokens if token]
    if not tokens:
        return []

    pieces = quote.split()
    pattern = tokens[0]
    for previous, current, escaped in zip(pieces, pieces[1:], tokens[1:]):
        separator = r"\s+" if previous[-1].isalnum() and current[0].isalnum() else r"\s*"
        pattern += separator + escaped
    matches: List[Tuple[float, float, int, int]] = []
    for match in re.compile(pattern, re.IGNORECASE).finditer(text):
        first = char_to_word[match.start()]
        last = char_to_word[match.end() - 1]
        if first < first_word or last > last_word:
            continue
        prefix = text[word_bounds[first][0]:match.start()]
        suffix = text[match.end():word_bounds[last][1]]
        if any(char.isalnum() for char in prefix + suffix):
            continue
        matches.append(
            (float(words[first]["start"]), float(words[last]["end"]), first, last)
        )
    return matches


def align_quote(
    words: Sequence[Dict],
    quote: str,
    *,
    first_word: int = 0,
    last_word: Optional[int] = None,
) -> Optional[Tuple[float, float, int, int]]:
    """First whole-word quote match inside an optional range, or ``None``."""
    matches = align_quote_matches(
        words, quote, first_word=first_word, last_word=last_word
    )
    return matches[0] if matches else None


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
