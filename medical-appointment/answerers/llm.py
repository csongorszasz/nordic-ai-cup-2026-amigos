"""LLM answerer: local instruct model, transcript-only, quote-cited spans.

Serves the **L1** prompt (whole transcript + few-shot), the best-measured rung
(T036/T038). Few-shot examples are drawn from the other supplied conversations
(LOCO-safe: the input's own conversation is excluded). Heavy model imports stay
lazy; a ``StubClient`` can be injected for tests.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .align import align_quote
from .base import Answer
from .llm_client import HFClient
from .llm_parse import parse_answers
from .llm_prompt import build_few_shot, build_l1_messages, qid_for

logger = logging.getLogger(__name__)

USE_FEW_SHOT = os.environ.get("MEDAPP_LLM_FEWSHOT", "1") != "0"

_few_shot_cache: Dict[str, Sequence[Tuple[str, str]]] = {}
_data_cache: Dict[str, object] = {}


def _transcript_id(transcript: Dict) -> str:
    name = os.path.basename(transcript.get("audio_filename") or "")
    if name.startswith("conversation_"):
        name = name[len("conversation_"):]
    if name.endswith(".mp3"):
        name = name[:-4]
    return name


def _build_few_shot(exclude_tid: str):
    """Balanced few-shot turns from conversations other than ``exclude_tid``."""
    if not USE_FEW_SHOT:
        return ()
    if exclude_tid in _few_shot_cache:
        return _few_shot_cache[exclude_tid]
    try:
        from . import modernbert_data as data

        if not _data_cache:
            rows_by_tid: Dict[str, list] = defaultdict(list)
            for row in data.load_rows():
                rows_by_tid[row["transcript_id"]].append(row)
            transcripts = {
                tid: data.load_transcript(tid) for tid in rows_by_tid
            }
            _data_cache.update(
                {
                    "rows_by_tid": rows_by_tid,
                    "transcripts": transcripts,
                    "evidence": data.load_evidence(),
                }
            )
        examples = build_few_shot(
            _data_cache["rows_by_tid"],
            _data_cache["transcripts"],
            _data_cache["evidence"],
            exclude_tid=exclude_tid,
        )
    except Exception:
        logger.exception("few-shot build failed; falling back to zero-shot.")
        examples = ()
    _few_shot_cache[exclude_tid] = examples
    return examples


class LLMAnswerer:
    name = "llm"
    supports_info = True

    def __init__(self, client=None, few_shot=None, refiner=None) -> None:
        self._client = client
        self._few_shot = few_shot  # None = build lazily from training data
        self._refiner = refiner  # None + unset env = no refinement
        self._refiner_ready = refiner is not None

    def _ensure_client(self):
        if self._client is None:
            self._client = HFClient()
        return self._client

    def _ensure_refiner(self):
        """Boundary refiner when ``MEDAPP_LLM_REFINER`` is set, else None."""
        if not self._refiner_ready:
            self._refiner_ready = True
            checkpoint = os.environ.get("MEDAPP_LLM_REFINER")
            if checkpoint:
                from .refiner import BoundaryRefiner

                self._refiner = BoundaryRefiner(checkpoint)
        return self._refiner

    def warm_up(self) -> None:
        warm = getattr(self._ensure_client(), "warm_up", None)
        if warm is not None:
            warm()
        refiner = self._ensure_refiner()
        if refiner is not None:
            refiner.warm_up()

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

        few_shot = (
            self._few_shot
            if self._few_shot is not None
            else _build_few_shot(_transcript_id(transcript))
        )

        client = self._ensure_client()
        messages = build_l1_messages(transcript, questions, few_shot)
        try:
            raw = client.generate(messages)
        except Exception:
            logger.exception("LLM generation failed; guessing no.")
            return [self._wrap(False, None, None, return_info) for _ in questions]

        expected = [qid_for(index) for index in range(len(questions))]
        parsed = parse_answers(raw, expected)
        refiner = self._ensure_refiner()

        results: List[Answer] = []
        for index, qid in enumerate(expected):
            entry = parsed.get(qid)
            if entry is None or entry.get("answer") is None:
                results.append(self._wrap(False, None, {"decided_by": "parse"}, return_info))
                continue
            if not entry["answer"]:
                results.append(self._wrap(False, None, {"decided_by": "no"}, return_info))
                continue
            aligned = align_quote(words, entry.get("quote") or "")
            if aligned is None:
                results.append(
                    self._wrap(True, None,
                               {"decided_by": "yes", "quote": entry.get("quote")},
                               return_info)
                )
                continue
            span = (aligned[0], aligned[1])
            info = {"decided_by": "yes", "quote": entry.get("quote")}
            if refiner is not None:
                try:
                    refined = refiner.refine(questions[index], words, aligned[2], aligned[3])
                except Exception:
                    logger.exception("refiner failed; using the LLM span.")
                    refined = None
                if refined is not None:
                    info["llm_span"] = list(span)
                    span = refined
            results.append(self._wrap(True, span, info, return_info))
        return results

    @staticmethod
    def _wrap(is_true: bool, span, info, return_info: bool):
        if return_info:
            return (bool(is_true), span, info or {})
        return (bool(is_true), span)


_ANSWERER: Optional[LLMAnswerer] = None


def build_answerer() -> LLMAnswerer:
    """Return a process-wide singleton.

    Unlike the legacy/modernbert answerers (whose models are module-level), the
    LLM client lives on the instance, so a fresh answerer per request would
    reload the ~16 GB model every time. ``example.py`` warms up one instance and
    then calls the factory again; both must share it.
    """
    global _ANSWERER
    if _ANSWERER is None:
        _ANSWERER = LLMAnswerer()
    return _ANSWERER
