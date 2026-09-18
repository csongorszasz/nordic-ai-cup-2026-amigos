"""LLM answerer: local instruct model, transcript-only, quote-cited spans.

Serves the **L1** prompt (whole transcript + few-shot), the best-measured rung
(T036/T038). Few-shot examples are drawn from the other supplied conversations
(LOCO-safe: the input's own conversation is excluded). Heavy model imports stay
lazy; a ``StubClient`` can be injected for tests.

Serving knobs (defaults reproduce the validated T039 behaviour unless noted):

* ``MEDAPP_LLM_FEWSHOT_COUNTS``  ``positive,refute,off_topic`` examples (``1,1,1``)
* ``MEDAPP_LLM_CITE_SEGMENT``    ask for the quoted segment id and align the
  quote to that occurrence (``0``; changes the prompt, so A/B it first)
* ``MEDAPP_ALIGN_FUZZY``         near-verbatim fallback ratio when the exact
  quote is not found (``0.85``; ``0`` disables). Only touches spans that would
  otherwise be ``None``.
* ``MEDAPP_SPAN_START_OFFSET`` / ``MEDAPP_SPAN_END_OFFSET``  seconds added to
  every span (``0``; fit with ``calibrate_spans.py``)
* ``MEDAPP_STRICT_STARTUP``      raise from ``warm_up`` if few-shot comes up
  empty instead of silently serving zero-shot (``0``; the serve job sets ``1``)

Anything that goes wrong (generation error, unparseable or truncated reply,
deadline) falls back per question to the question-text prior, never to a
constant.
"""

import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .align import align_span, apply_offsets
from .base import Answer
from .llm_client import HFClient
from .llm_parse import parse_answers
from .llm_prompt import build_few_shot, build_l1_messages, qid_for

logger = logging.getLogger(__name__)

USE_FEW_SHOT = os.environ.get("MEDAPP_LLM_FEWSHOT", "1") != "0"
# "4,1,1" or "4:1:1" (sbatch --export splits on commas, so use colons there).
FEW_SHOT_COUNTS = tuple(
    int(part)
    for part in re.split(r"[,:]", os.environ.get("MEDAPP_LLM_FEWSHOT_COUNTS", "1,1,1"))
)
CITE_SEGMENT = os.environ.get("MEDAPP_LLM_CITE_SEGMENT", "0") == "1"
FUZZY_RATIO = float(os.environ.get("MEDAPP_ALIGN_FUZZY", "0.85")) or None
START_OFFSET = float(os.environ.get("MEDAPP_SPAN_START_OFFSET", "0"))
END_OFFSET = float(os.environ.get("MEDAPP_SPAN_END_OFFSET", "0"))
STRICT = os.environ.get("MEDAPP_STRICT_STARTUP", "0") == "1"
# Seconds kept back from the deadline for parsing, alignment and the response.
RESERVE_S = float(os.environ.get("MEDAPP_LLM_RESERVE_S", "2"))
# Below this much remaining time a generation is not worth starting.
MIN_GENERATION_S = float(os.environ.get("MEDAPP_LLM_MIN_GENERATION_S", "4"))

# The id used for few-shot selection when serving an unseen conversation. Any
# id that is not a training conversation keeps every example available.
_UNSEEN = "__unseen__"

_few_shot_cache: Dict[tuple, Sequence[Tuple[str, str]]] = {}
_data_cache: Dict[str, object] = {}


def _transcript_id(transcript: Dict) -> str:
    name = os.path.basename(transcript.get("audio_filename") or "")
    if name.startswith("conversation_"):
        name = name[len("conversation_"):]
    if name.endswith(".mp3"):
        name = name[:-4]
    return name


def _build_few_shot(
    exclude_tid: str,
    counts: Tuple[int, int, int] = FEW_SHOT_COUNTS,
    cite_segment: bool = CITE_SEGMENT,
    strict: bool = False,
):
    """Balanced few-shot turns from conversations other than ``exclude_tid``.

    With ``strict`` a failure raises instead of degrading to zero-shot (which
    costs ~0.08 in-sample); the serving warm-up uses it so a missing
    ``transcripts/`` directory stops the deploy rather than the score.
    """
    if not USE_FEW_SHOT:
        return ()
    key = (exclude_tid, tuple(counts), cite_segment)
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
            counts=tuple(counts),
            cite_segment=cite_segment,
        )
    except Exception:
        if strict:
            raise
        logger.exception("few-shot build failed; falling back to zero-shot.")
        examples = ()
    if strict and not examples:
        raise RuntimeError(
            "few-shot came up empty: are data/, annotations/ and transcripts/ present?"
        )
    _few_shot_cache[key] = examples
    return examples


def _prior_guesses(questions: Sequence[str]) -> List[bool]:
    from prior import guess

    return guess(questions)


class LLMAnswerer:
    name = "llm"
    supports_info = True

    def __init__(
        self,
        client=None,
        few_shot=None,
        fallback=None,
        cite_segment: bool = CITE_SEGMENT,
        fuzzy_ratio: Optional[float] = FUZZY_RATIO,
        offsets: Tuple[float, float] = (START_OFFSET, END_OFFSET),
        few_shot_counts: Tuple[int, int, int] = FEW_SHOT_COUNTS,
    ) -> None:
        self._client = client
        self._few_shot = few_shot  # None = build lazily from training data
        # questions -> List[bool]; the guess for anything the LLM did not answer.
        self._fallback = fallback or _prior_guesses
        self.cite_segment = cite_segment
        self.fuzzy_ratio = fuzzy_ratio
        self.offsets = offsets
        self.few_shot_counts = tuple(few_shot_counts)
        self.few_shot_size = None

    def _ensure_client(self):
        if self._client is None:
            self._client = HFClient()
        return self._client

    def warm_up(self) -> None:
        """Load the model, the few-shot examples and the prior before serving."""
        warm = getattr(self._ensure_client(), "warm_up", None)
        if warm is not None:
            warm()
        if self._few_shot is None:
            examples = _build_few_shot(
                _UNSEEN, counts=self.few_shot_counts, cite_segment=self.cite_segment,
                strict=STRICT,
            )
            self.few_shot_size = len(examples)
        self._fallback(["Is the endpoint warm?"])

    def status(self) -> Dict:
        client = self._client
        return {
            "answerer": self.name,
            "model": getattr(client, "model_name", None),
            "revision": getattr(client, "revision", None),
            "few_shot_examples": self.few_shot_size,
            "few_shot_counts": list(self.few_shot_counts),
            "cite_segment": self.cite_segment,
            "fuzzy_ratio": self.fuzzy_ratio,
            "offsets": list(self.offsets),
        }

    def _guesses(self, questions, return_info, reason):
        try:
            answers = list(self._fallback(questions))
        except Exception:
            logger.exception("fallback guesses failed; using yes.")
            answers = []
        answers += [True] * (len(questions) - len(answers))
        return [
            self._wrap(answer, None, {"decided_by": reason}, return_info)
            for answer in answers[: len(questions)]
        ]

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
            return self._guesses(questions, return_info, "empty_transcript")

        max_time = None
        if deadline is not None:
            max_time = deadline - time.time() - RESERVE_S
            if max_time < MIN_GENERATION_S:
                logger.warning("%.1fs left before the deadline; guessing.", max_time)
                return self._guesses(questions, return_info, "deadline")

        few_shot = (
            self._few_shot
            if self._few_shot is not None
            else _build_few_shot(
                _transcript_id(transcript), counts=self.few_shot_counts,
                cite_segment=self.cite_segment,
            )
        )

        client = self._ensure_client()
        messages = build_l1_messages(
            transcript, questions, few_shot, cite_segment=self.cite_segment
        )
        try:
            raw = client.generate(messages, max_time=max_time)
        except Exception:
            logger.exception("LLM generation failed; guessing.")
            return self._guesses(questions, return_info, "generation_error")

        expected = [qid_for(index) for index in range(len(questions))]
        parsed = parse_answers(raw, expected)
        missing = [index for index, qid in enumerate(expected)
                   if (parsed.get(qid) or {}).get("answer") is None]
        if missing:
            logger.warning(
                "LLM reply left %d/%d questions unanswered (len=%d, tail=%r); guessing those.",
                len(missing), len(expected), len(raw or ""), (raw or "")[-120:],
            )
        guesses = (
            dict(zip(missing, self._guesses(
                [questions[index] for index in missing], return_info, "parse"
            )))
            if missing else {}
        )

        duration = transcript.get("duration")
        results: List[Answer] = []
        for index, qid in enumerate(expected):
            if index in guesses:
                results.append(guesses[index])
                continue
            entry = parsed[qid]
            if not entry["answer"]:
                results.append(self._wrap(False, None, {"decided_by": "no"}, return_info))
                continue
            segment = entry.get("segment") if self.cite_segment else None
            span = align_span(
                words, entry.get("quote") or "", segment=segment,
                fuzzy_ratio=self.fuzzy_ratio,
            )
            span = apply_offsets(span, *self.offsets, duration=duration)
            results.append(
                self._wrap(True, span, {"decided_by": "yes", "quote": entry.get("quote"),
                                        "segment": segment}, return_info)
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
