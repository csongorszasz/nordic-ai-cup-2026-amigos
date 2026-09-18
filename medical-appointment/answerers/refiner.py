"""Post-hoc boundary refiner for LLM cite spans.

The LLM decides yes/no and quotes the evidence; the quote is aligned to the ASR
word stream, but the *extent* of the quoted span is often not the extent a human
annotator marked (too long, too short, or shifted). This module refines that
extent with a small encoder trained on the 39 supplied conversations: the input
is the question plus a window of words around the LLM's occurrence, **with the
quoted words marked in place**, and the output is a pair of word adjustments
(``left``/``right``) applied to the anchor. Predicting the edit makes "keep the
LLM span" the zero default, so a correct quote is never moved elsewhere.

The decision is never changed — only the span of an answered yes. Every failure
returns ``None`` so the caller can fall back to the LLM span, and the heavy
torch imports stay lazy so the module is import-safe in the torch-free envs.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from windows import join_words

from .passages import Passage
from .span_utils import char_span_to_words, token_span_to_char, word_char_spans

logger = logging.getLogger(__name__)

WINDOW = int(os.environ.get("MEDAPP_REFINER_WINDOW", "24"))
MAX_LENGTH = int(os.environ.get("MEDAPP_REFINER_MAX_LENGTH", "256"))
MAX_WORDS = int(os.environ.get("MEDAPP_REFINER_MAX_WORDS", "40"))
CHECKPOINT_ENV = "MEDAPP_LLM_REFINER"

EVIDENCE_OPEN = "[EVIDENCE]"
EVIDENCE_CLOSE = "[/EVIDENCE]"


def window_bounds(
    n_words: int, first: int, last: int, context: int = WINDOW
) -> Tuple[int, int]:
    """Word index range ``context`` words either side of ``[first, last]``."""
    lo = max(0, first - context)
    hi = min(n_words - 1, last + context)
    return lo, hi


def build_window_passage(
    words: Sequence[Dict], first: int, last: int, context: int = WINDOW
) -> Optional[Passage]:
    """The window around an aligned word range, as a ``passage``-style unit."""
    if not words:
        return None
    lo, hi = window_bounds(len(words), first, last, context)
    if lo > hi:
        return None
    return Passage(
        index=0,
        first_word=lo,
        last_word=hi,
        start=float(words[lo]["start"]),
        end=float(words[hi]["end"]),
        text=join_words(words, lo, hi),
    )


def marked_window_layout(
    passage_words: Sequence[str], anchor_first: int, anchor_last: int
) -> Tuple[str, List[Tuple[int, int]]]:
    """Window text with the LLM quote marked, plus each word's char span.

    The markers are separate tokens; the returned spans line up with
    ``passage_words`` (not with the token list), so word targets are unaffected.
    """
    tokens: List[Tuple[str, bool]] = []
    for index, word in enumerate(passage_words):
        if index == anchor_first:
            tokens.append((EVIDENCE_OPEN, False))
        if index == anchor_last + 1:
            tokens.append((EVIDENCE_CLOSE, False))
        tokens.append((word, True))
    if passage_words and anchor_last + 1 >= len(passage_words):
        tokens.append((EVIDENCE_CLOSE, False))

    pieces: List[str] = []
    spans: List[Tuple[int, int]] = []
    position = 0
    for token, is_word in tokens:
        if pieces:
            position += 1
        if is_word:
            spans.append((position, position + len(token)))
        pieces.append(token)
        position += len(token)
    return " ".join(pieces), spans


def relative_word_span_from_spans(
    word_spans: Sequence[Tuple[int, int]],
    offsets: Sequence,
    sequence_ids: Sequence,
    start_index: int,
    end_index: int,
) -> Optional[Tuple[int, int]]:
    """Predicted token span -> word indices, given precomputed word char spans."""
    char = token_span_to_char(offsets, sequence_ids, start_index, end_index)
    if char is None:
        return None
    return char_span_to_words(word_spans, char[0], char[1])


def relative_word_span(
    passage_words: Sequence[str],
    offsets: Sequence,
    sequence_ids: Sequence,
    start_index: int,
    end_index: int,
) -> Optional[Tuple[int, int]]:
    """Predicted token span -> word indices relative to the passage."""
    _, word_spans = word_char_spans(passage_words)
    return relative_word_span_from_spans(
        word_spans, offsets, sequence_ids, start_index, end_index
    )


def joint_token_span(
    start_scores: Sequence[float],
    end_scores: Sequence[float],
    sequence_ids: Sequence,
    max_words: int = MAX_WORDS,
) -> Optional[Tuple[int, int]]:
    """Best passage ``(start, end)`` by ``start[a] + end[b]`` with ``b >= a``.

    A joint argmax stops the decoder pairing the best start with an unrelated
    best end; ``max_words`` caps the span length in words.
    """
    passage = [index for index, seq in enumerate(sequence_ids) if seq == 1]
    if not passage:
        return None
    best: Optional[Tuple[int, int]] = None
    best_score = float("-inf")
    for start in passage:
        for end in passage:
            if end < start or end - start >= max_words:
                continue
            score = float(start_scores[start]) + float(end_scores[end])
            if score > best_score:
                best_score = score
                best = (start, end)
    return best


def apply_deltas(
    first_word: int,
    last_word: int,
    left: float,
    right: float,
    low: int,
    high: int,
) -> Tuple[int, int]:
    """Anchor word range adjusted by rounded deltas, clamped to ``[low, high]``."""
    new_first = first_word - int(round(left))
    new_last = last_word + int(round(right))
    new_first = max(low, min(high, new_first))
    new_last = max(low, min(high, new_last))
    new_last = max(new_first, new_last)
    return new_first, new_last


class BoundaryRefiner:
    """Trained delta refiner; constructed only when a checkpoint is given."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        window: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.checkpoint = checkpoint or os.environ.get(CHECKPOINT_ENV)
        self.window = WINDOW if window is None else window
        self.max_length = MAX_LENGTH if max_length is None else max_length
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._module = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from . import refiner_model as module

        self._module = module
        model, device = module.load_delta_refiner(self.checkpoint)
        tokenizer = module.AutoTokenizer.from_pretrained(module.MODEL_NAME)
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def warm_up(self) -> None:
        if not self.checkpoint:
            raise RuntimeError(
                "MEDAPP_LLM_REFINER is not set; refusing to load the refiner."
            )
        self._load()

    def refine(
        self, question: str, words: Sequence[Dict], first_word: int, last_word: int
    ) -> Optional[Tuple[float, float]]:
        """Refined ``(start, end)`` seconds around the LLM's occurrence, or None."""
        if not self.checkpoint:
            return None
        passage = build_window_passage(words, first_word, last_word, self.window)
        if passage is None:
            return None
        self._load()
        passage_words = [
            words[k]["word"].strip()
            for k in range(passage.first_word, passage.last_word + 1)
        ]
        text, _ = marked_window_layout(
            passage_words,
            first_word - passage.first_word,
            last_word - passage.first_word,
        )
        encoding = self._tokenizer(
            question,
            text,
            truncation="only_second",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {key: value.to(self._device) for key, value in encoding.items()}
        import torch

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
        first, last = apply_deltas(
            first_word,
            last_word,
            float(outputs["left"][0]),
            float(outputs["right"][0]),
            passage.first_word,
            passage.last_word,
        )
        return float(words[first]["start"]), float(words[last]["end"])
