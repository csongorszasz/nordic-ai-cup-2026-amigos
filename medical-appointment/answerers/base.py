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

``deadline`` is an absolute wall-clock timestamp (``time.time()`` epoch), not a
duration: an implementation that senses it has passed must stop doing expensive
work and return a well-formed guess for the remaining questions. This mirrors
the served deadline fallback in ``example.py``.
"""

from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

# A span of audio in seconds from the start of the conversation.
Span = Tuple[float, float]

# One answer: the boolean and, for a yes, the evidence span.
Answer = Tuple[bool, Optional[Span]]


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

    Used by an answerer that hit its deadline: ``True`` with no span, which is
    the same floor guess ``example.py`` falls back to.
    """
    return [(True, None)] * max(0, len(questions) - count)
