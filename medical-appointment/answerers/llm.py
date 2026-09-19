"""LLM answerer: local instruct model, transcript-only, quote-cited spans.

Serves the **L1** prompt (whole transcript + few-shot), the best-measured rung
(T036/T038). Few-shot examples are drawn from the other supplied conversations
(LOCO-safe: the input's own conversation is excluded). Heavy model imports stay
lazy; a ``StubClient`` can be injected for tests.
"""

import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .align import align_quote
from .base import Answer
from .boundaries import OffsetCalibration
from .llm_client import HFClient
from .llm_parse import parse_answers
from .llm_prompt import (
    VARIANT, build_few_shot, build_l1_messages, few_shot_counts, qid_for,
    select_few_shot_rows, system_prompt,
)

logger = logging.getLogger(__name__)

USE_FEW_SHOT = os.environ.get("MEDAPP_LLM_FEWSHOT", "1") != "0"

_few_shot_cache: Dict[Tuple[str, str], Sequence[Tuple[str, str]]] = {}
_data_cache: Dict[str, object] = {}
_few_shot_sources: Dict[Tuple[str, str], List[str]] = {}


def _transcript_id(transcript: Dict) -> str:
    name = os.path.basename(transcript.get("audio_filename") or "")
    if name.startswith("conversation_"):
        name = name[len("conversation_"):]
    if name.endswith(".mp3"):
        name = name[:-4]
    return name


def _build_few_shot(exclude_tid: str, variant: str = VARIANT):
    """Balanced few-shot turns from conversations other than ``exclude_tid``."""
    if not USE_FEW_SHOT:
        return ()
    key = (exclude_tid, variant)
    if key in _few_shot_cache:
        return _few_shot_cache[key]
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
            variant=variant,
        )
        _few_shot_sources[key] = [
            row["transcript_id"]
            for row in select_few_shot_rows(
                _data_cache["rows_by_tid"], _data_cache["transcripts"],
                _data_cache["evidence"], exclude_tid, few_shot_counts(variant),
            )
        ]
    except Exception:
        logger.exception("Few-shot build failed; refusing an unmeasured zero-shot fallback.")
        raise
    _few_shot_cache[key] = examples
    return examples


def evidence_region(transcript: Dict, entry: Dict) -> Optional[Tuple[int, int]]:
    """Resolve cited segment IDs to a contiguous word range."""
    start, end = entry.get("segment_start"), entry.get("segment_end")
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    if not re.fullmatch(r"s\d+", start) or not re.fullmatch(r"s\d+", end):
        return None
    positions = {int(segment["id"]): index for index, segment in enumerate(transcript["segments"])}
    first_seg, last_seg = positions.get(int(start[1:])), positions.get(int(end[1:]))
    if first_seg is None or last_seg is None or first_seg > last_seg:
        return None
    indices = [
        index for index, word in enumerate(transcript["words"])
        if first_seg <= word.get("seg_idx", -1) <= last_seg
    ]
    return (indices[0], indices[-1]) if indices else None


class LLMAnswerer:
    name = "llm"
    supports_info = True

    def __init__(
        self, client=None, few_shot=None, variant: str = VARIANT,
        calibration: Optional[OffsetCalibration] = None,
    ) -> None:
        self._client = client
        self._few_shot = few_shot  # None = build lazily from training data
        self.variant = variant.strip().lower()
        system_prompt(self.variant)
        self.few_shot_sources: List[str] = []
        calibration_path = os.environ.get("MEDAPP_SPAN_CALIBRATION")
        self.calibration = (
            calibration if calibration is not None else
            OffsetCalibration.load(calibration_path) if calibration_path else None
        )

    def _ensure_client(self):
        if self._client is None:
            self._client = HFClient()
        return self._client

    def warm_up(self) -> None:
        self._validate_calibration()
        if self._few_shot is None:
            _build_few_shot("", self.variant)
        warm = getattr(self._ensure_client(), "warm_up", None)
        if warm is not None:
            warm()

    def _validate_calibration(self, transcript=None) -> None:
        if self.calibration is not None:
            import asr

            client = self._ensure_client()
            if transcript is not None and (
                transcript.get("model", asr.MODEL_SIZE) != asr.MODEL_SIZE
                or transcript.get("_cache_config_hash", asr.config_hash()) != asr.config_hash()
            ):
                raise ValueError("Transcript provenance does not match the boundary calibration.")
            self.calibration.validate_context(
                asr_config_hash=asr.config_hash(),
                model=client.model_name, revision=client.revision, variant=self.variant,
            )

    def answer_all(
        self,
        questions: Sequence[str],
        transcript: Dict,
        *,
        deadline: Optional[float] = None,
        return_info: bool = False,
    ) -> List[Answer]:
        self.few_shot_sources = []
        words: List[Dict] = transcript.get("words", [])
        if not words:
            return [
                self._wrap(False, None, {"decided_by": "empty_transcript"}, return_info)
                for _ in questions
            ]

        if deadline is not None and time.monotonic() >= deadline:
            logger.warning("LLM deadline expired; returning no/null guesses.")
            return [
                self._wrap(False, None, {"decided_by": "deadline"}, return_info)
                for _ in questions
            ]

        few_shot = (
            self._few_shot
            if self._few_shot is not None
            else _build_few_shot(_transcript_id(transcript), self.variant)
        )
        self.few_shot_sources = list(
            _few_shot_sources.get((_transcript_id(transcript), self.variant), ())
        ) if self._few_shot is None else []

        client = self._ensure_client()
        self._validate_calibration(transcript)
        messages = build_l1_messages(transcript, questions, few_shot, self.variant)
        try:
            raw = (
                client.generate(messages, deadline=deadline)
                if deadline is not None else client.generate(messages)
            )
        except Exception:
            logger.exception("LLM generation failed; guessing no.")
            return [
                self._wrap(False, None, {"decided_by": "generation"}, return_info)
                for _ in questions
            ]

        expected = [qid_for(index) for index in range(len(questions))]
        parsed = parse_answers(raw, expected)

        results: List[Answer] = []
        for qid in expected:
            entry = parsed.get(qid)
            if entry is None or entry.get("answer") is None:
                logger.warning("Unusable LLM answer for %s; guessing no.", qid)
                results.append(self._wrap(False, None, {"decided_by": "parse"}, return_info))
                continue
            if not entry["answer"]:
                results.append(
                    self._wrap(False, None, {"decided_by": "no", "raw_answer": False}, return_info)
                )
                continue
            region = evidence_region(transcript, entry) if self.variant == "scoped" else None
            aligned = (
                align_quote(
                    words, entry.get("quote") or "",
                    **({"first_word": region[0], "last_word": region[1]} if region else {}),
                )
                if self.variant != "scoped" or region is not None else None
            )
            span = (aligned[0], aligned[1]) if aligned is not None else None
            original_span = span
            if span is not None and self.calibration is not None:
                span = self.calibration.apply(span, transcript.get("duration"))
            if span is None:
                logger.warning("Cannot ground %s's evidence quote; guessing no.", qid)
            results.append(
                self._wrap(
                    span is not None, span,
                    {
                        "decided_by": "yes" if span is not None else "alignment",
                        "raw_answer": True,
                        "quote": entry.get("quote"),
                        "region": region,
                        "word_range": list(aligned[2:]) if aligned is not None else None,
                        "original_span": original_span,
                        "calibration_applied": span != original_span,
                    },
                    return_info,
                )
            )
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
