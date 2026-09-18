"""LLM answerer: local instruct model, transcript-only, quote-cited spans.

This is the probe answerer behind ``MEDAPP_ANSWERER=llm`` (L0-style prompt). The
heavy model loads lazily and can be injected as a ``StubClient`` for tests. It
does not add few-shot examples by default; ``llm_probe.py`` drives the L1/L2
variants directly.
"""

import logging
import time
from typing import Dict, List, Optional, Sequence

from .align import align_span
from .base import Answer
from .llm_client import HFClient
from .llm_parse import parse_answers
from .llm_prompt import build_l0_messages, qid_for

logger = logging.getLogger(__name__)


class LLMAnswerer:
    name = "llm"
    supports_info = True

    def __init__(self, client=None) -> None:
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            self._client = HFClient()
        return self._client

    def warm_up(self) -> None:
        warm = getattr(self._ensure_client(), "warm_up", None)
        if warm is not None:
            warm()

    def answer_all(
        self,
        questions: Sequence[str],
        transcript: Dict,
        *,
        deadline: Optional[float] = None,
        return_info: bool = False,
    ) -> List[Answer]:
        words: List[Dict] = transcript.get("words", [])
        if not words:
            return [self._wrap(False, None, None, return_info) for _ in questions]

        if deadline is not None and time.time() > deadline:
            return [self._wrap(True, None, None, return_info) for _ in questions]

        client = self._ensure_client()
        messages = build_l0_messages(transcript, questions)
        try:
            raw = client.generate(messages)
        except Exception:
            logger.exception("LLM generation failed; guessing no.")
            return [self._wrap(False, None, None, return_info) for _ in questions]

        expected = [qid_for(index) for index in range(len(questions))]
        parsed = parse_answers(raw, expected)

        results: List[Answer] = []
        for qid in expected:
            entry = parsed.get(qid)
            if entry is None or entry.get("answer") is None:
                results.append(self._wrap(False, None, {"decided_by": "parse"}, return_info))
                continue
            if not entry["answer"]:
                results.append(self._wrap(False, None, {"decided_by": "no"}, return_info))
                continue
            span = align_span(words, entry.get("quote") or "")
            results.append(
                self._wrap(True, span, {"decided_by": "yes", "quote": entry.get("quote")},
                           return_info)
            )
        return results

    @staticmethod
    def _wrap(is_true: bool, span, info, return_info: bool):
        if return_info:
            return (bool(is_true), span, info or {})
        return (bool(is_true), span)


def build_answerer() -> LLMAnswerer:
    return LLMAnswerer()
