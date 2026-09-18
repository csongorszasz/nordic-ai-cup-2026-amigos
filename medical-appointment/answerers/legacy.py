"""The frozen NLI answerer: a thin adapter over ``answer.py``.

This wraps the existing pipeline exactly as ``example.py`` used to call it —
build phrase-level windows once, then answer each question with
``answer.answer_question`` — so the served default keeps reproducing the
validated numbers (T028 base 0.541 / T029 large 0.566). ``answer.py`` and
``windows.py`` are treated as frozen; do not add behaviour here.

The call to ``answer.answer_question`` is resolved as a module attribute at
call time, so tests that monkeypatch ``answer.answer_question`` still take
effect through this wrapper.
"""

import logging
import time
from typing import Dict, List, Optional, Sequence

import answer as answer_module
import windows as windows_module

from .base import Answer, Span, with_deadline_guesses

logger = logging.getLogger(__name__)


class LegacyAnswerer:
    """Adapter from the ``Answerer`` protocol to ``answer.py``."""

    name = "legacy"

    def warm_up(self) -> None:
        """Load the NLI model now so the first request is not the slowest."""
        answer_module.nli.warm_up()

    def answer_all(
        self,
        questions: Sequence[str],
        transcript: Dict,
        *,
        deadline: Optional[float] = None,
    ) -> List[Answer]:
        words: List[Dict] = transcript.get("words", [])
        windows = windows_module.windows_from_transcript(transcript)

        answers: List[Answer] = []
        for question in questions:
            # Budget guard: a valid guess beats blowing the 60 s request budget.
            if deadline is not None and time.time() > deadline:
                logger.warning(
                    "Deadline reached; guessing remaining %d questions.",
                    len(questions) - len(answers),
                )
                answers.extend(with_deadline_guesses(questions, len(answers)))
                break

            try:
                is_true, span = answer_module.answer_question(
                    question, words, windows
                )
            except Exception:
                logger.exception("Answering failed; guessing for: %s", question)
                is_true, span = True, None

            answers.append((bool(is_true), span))

        return answers


def build_answerer() -> LegacyAnswerer:
    return LegacyAnswerer()
