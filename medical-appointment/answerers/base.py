"""The answerer contract shared by every implementation.

An *answerer* turns the ten yes/no questions about one conversation into
``(bool, span | None)`` pairs, where ``span`` is the ``(start, end)`` evidence
interval in seconds. The transcript dict is the one produced by
:func:`asr.transcribe_bytes` (segments plus a flat word list with timestamps).

The protocol is deliberately narrow so implementations can differ wildly
underneath — the frozen NLI pipeline and a task-trained cross-encoder both fit:

    answerer = get_answerer()
    answers = answerer.answer_all(questions, transcript)

``answer_all`` returns exactly one entry per question, in order. A ``False``
answer must carry ``None``; a ``True`` answer should carry a span, but a span is
allowed to be ``None`` (the scorer treats it as tIoU 0 rather than raising).

``deadline`` is an absolute monotonic timestamp (``time.monotonic()``), not a
duration: an implementation that senses it has passed must stop doing expensive
work and return a well-formed guess for the remaining questions. This mirrors
the served deadline fallback in ``example.py``.
"""

import logging
import math
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

# A span of audio in seconds from the start of the conversation.
Span = Tuple[float, float]

# One answer: the boolean and, for a yes, the evidence span.
Answer = Tuple[bool, Optional[Span]]


def normalize_answer(entry, *, duration=None, context="answer") -> Answer:
    """Apply the same per-slot evidence/fallback contract in serving and scoring."""
    try:
        if not isinstance(entry, (tuple, list)) or len(entry) not in (2, 3):
            raise ValueError("Missing or malformed answer slot.")
        is_true, span = entry[:2]
        if not isinstance(is_true, bool):
            raise ValueError("Answer must be a boolean.")
        if not is_true:
            if span is not None:
                logger.warning("%s: discarding evidence attached to a no answer.", context)
            return False, None
        if span is None or len(span) != 2:
            raise ValueError("Positive answer has no complete evidence span.")
        start, end = span
        if (
            any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in (start, end))
            or start < 0 or end <= start
            or (duration is not None and end > duration)
        ):
            raise ValueError("Evidence is not a finite interval within the audio.")
        return True, (float(start), float(end))
    except (TypeError, ValueError) as exc:
        logger.warning("%s: %s Returning a no/null guess.", context, exc)
        return False, None


@runtime_checkable
class Answerer(Protocol):
    """Answers a batch of questions about a single conversation."""

    def answer_all(
        self,
        questions: Sequence[str],
        transcript: Dict,
        *,
        deadline: Optional[float] = None,
    ) -> List[Answer]:
        ...  # pragma: no cover - protocol declaration


def with_deadline_guesses(
    questions: Sequence[str], count: int
) -> List[Answer]:
    """Well-formed guesses for the questions not yet answered.

    A no/null guess satisfies the evidence contract without inventing a span.
    """
    return [(False, None)] * max(0, len(questions) - count)
