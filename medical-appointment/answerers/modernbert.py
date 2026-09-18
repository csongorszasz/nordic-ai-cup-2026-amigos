"""The ModernBERT candidate-scoring answerer.

Per conversation: build sliding passages, retrieve the top-k per question with
MiniLM, score each ``(question, passage)`` pair with the ModernBERT cross-encoder
(:mod:`answerers.modernbert_model`), then decode.

Decode rule (score-aware): among the candidates the model classifies SUPPORT,
take the one with the highest predicted tIoU; answer ``yes`` iff

    p_support > 0.4 / (0.8 + 1.2 * expected_tiou)

and return that candidate's predicted token span mapped back to word timestamps.

The heavy imports are lazy: this module is import-safe without torch, so the
factory and its tests work in the torch-free ``medical`` env. Selecting
``MEDAPP_ANSWERER=modernbert`` without a checkpoint raises a clear error.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .base import Answer
from .minilm import MiniLMRetriever
from .modernbert_data import TOP_K
from .passages import build_passages
from .span_utils import (
    pick_supported_candidate,
    psupport_accept,
    score_aware_accept,
)

logger = logging.getLogger(__name__)

CHECKPOINT = os.environ.get("MEDAPP_MB_CHECKPOINT") or None
# Decoder: "psupport" (max p_support, thresholded -- the OOF winner) or
# "score_aware" (argmax SUPPORT, q-dependent cutoff). tau is LOCO-calibrated:
# on the class-weighted OOF the LOCO optimum was 0.16 (stable 0.15-0.16), with
# a flat plateau from 0.05 to 0.16.
DECODE_MODE = os.environ.get("MEDAPP_MB_DECODE", "psupport")
DECODE_TAU = float(os.environ.get("MEDAPP_MB_TAU", "0.16"))

_state: Dict[str, object] = {}


def _load():
    """Load the scorer, tokenizer and retriever once (lazily)."""
    if _state:
        return _state
    from . import modernbert_model as model_module

    scorer, device = model_module.load_scorer(CHECKPOINT)
    tokenizer = model_module.AutoTokenizer.from_pretrained(model_module.MODEL_NAME)
    retriever = MiniLMRetriever()
    _state.update(
        {"model": scorer, "tokenizer": tokenizer, "retriever": retriever,
         "device": device, "module": model_module}
    )
    return _state


def retrieve_candidates(
    questions: Sequence[str],
    passages,
    retriever: MiniLMRetriever,
    passage_embeddings,
    top_k: int = TOP_K,
):
    """Top-k passages per question, returned as ``(pairs, per_question_buckets)``.

    ``pairs`` is the flat list fed to the cross-encoder; each bucket maps a
    question to ``(passage, pair_index)`` entries.
    """
    pairs: List[Tuple[str, str]] = []
    per_question: List[List] = []
    for question in questions:
        ranked = retriever.rank(
            question, passages, top_k=top_k, passage_embeddings=passage_embeddings
        )
        bucket = []
        for passage, _ in ranked:
            bucket.append((passage, len(pairs)))
            pairs.append((question, passage.text))
        per_question.append(bucket)
    return pairs, per_question


def make_candidates(bucket, scored) -> List[Dict]:
    candidates = []
    for passage, pair_index in bucket:
        result = scored[pair_index]
        candidates.append(
            {
                "passage": passage,
                "result": result,
                "label": result["label"],
                "p_support": result["p_support"],
                "expected_tiou": result["expected_tiou"],
            }
        )
    return candidates


def decode_question(
    candidates: List[Dict],
    words: List[Dict],
    module,
    mode: Optional[str] = None,
    tau: Optional[float] = None,
):
    """Pick a candidate and decide yes/no.

    ``psupport`` (default) takes the candidate with the highest ``p_support``
    and accepts when it clears ``tau``. ``score_aware`` takes the argmax SUPPORT
    candidate and uses ``p > 0.4 / (0.8 + 1.2 q)``.

    Returns ``(answer, span, info)``. ``info['span']`` always carries the chosen
    candidate's predicted span (even when the answer is no) so out-of-fold
    calibration can sweep a threshold without re-running the model.
    """
    mode = mode or DECODE_MODE
    threshold = DECODE_TAU if tau is None else tau

    if not candidates:
        return False, None, {"decided_by": "no_candidates", "p": 0.0, "span": None}

    if mode == "score_aware":
        best = pick_supported_candidate(candidates)
        info_only = best is None
        if info_only:
            best = max(
                range(len(candidates)), key=lambda i: candidates[i]["p_support"]
            )
    else:  # psupport
        best = max(range(len(candidates)), key=lambda i: candidates[i]["p_support"])
        info_only = False

    chosen = candidates[best]
    passage = chosen["passage"]
    result = chosen["result"]
    span = module.predicted_span_seconds(
        passage, words, result["offsets"], result["sequence_ids"],
        result["start_index"], result["end_index"],
    )
    if span is None:
        span = passage.span()

    p_support = float(chosen["p_support"])
    expected_tiou = float(chosen["expected_tiou"])
    info = {
        "p": p_support,
        "expected_tiou": expected_tiou,
        "span": span,
        "mode": mode,
        "tau": threshold,
        "decided_by": "no_support" if info_only else "support",
    }

    if mode == "score_aware":
        if info_only:
            return False, None, info
        accepted = score_aware_accept(p_support, expected_tiou)
    else:
        accepted = psupport_accept(p_support, threshold)

    if accepted:
        return True, span, info
    info["decided_by"] = "below_threshold"
    return False, None, info


def predict_transcript(model, tokenizer, retriever, module, device, words, questions):
    """Answer every question about one already-transcribed conversation.

    Shared by serving and out-of-fold scoring so the two cannot drift.
    """
    passages = build_passages(words)
    passage_embeddings = retriever.encode([p.text for p in passages])
    pairs, per_question = retrieve_candidates(
        questions, passages, retriever, passage_embeddings
    )
    scored = module.score_pairs(model, tokenizer, pairs, device)
    results = []
    for bucket in per_question:
        results.append(
            decode_question(make_candidates(bucket, scored), words, module)
        )
    return results


class ModernBertAnswerer:
    """Adapter from the ``Answerer`` protocol to the trained scorer."""

    name = "modernbert"
    supports_info = True

    def warm_up(self) -> None:
        if CHECKPOINT is None:
            raise RuntimeError(
                "MEDAPP_MB_CHECKPOINT is not set; keep MEDAPP_ANSWERER=legacy."
            )
        state = _load()
        state["retriever"].warm_up()

    def answer_all(
        self,
        questions: Sequence[str],
        transcript: Dict,
        *,
        deadline: Optional[float] = None,
        return_info: bool = False,
    ) -> List[Answer]:
        if CHECKPOINT is None:
            raise NotImplementedError(
                "ModernBERT answerer has no checkpoint (set MEDAPP_MB_CHECKPOINT); "
                "keep MEDAPP_ANSWERER=legacy."
            )

        words: List[Dict] = transcript.get("words", [])
        if not words:
            return [self._wrap(False, None, None, return_info) for _ in questions]

        state = _load()

        if deadline is not None and time.time() > deadline:
            logger.warning("Deadline passed before ModernBERT scoring; guessing.")
            return [self._wrap(True, None, None, return_info) for _ in questions]

        results = predict_transcript(
            state["model"], state["tokenizer"], state["retriever"],
            state["module"], state["device"], words, questions,
        )
        return [
            self._wrap(answer, span, info, return_info)
            for answer, span, info in results
        ]

    @staticmethod
    def _wrap(is_true: bool, span, info, return_info: bool):
        if return_info:
            return (bool(is_true), span, info or {})
        return (bool(is_true), span)


def build_answerer() -> ModernBertAnswerer:
    return ModernBertAnswerer()
