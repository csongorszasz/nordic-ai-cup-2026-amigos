"""The ModernBERT candidate-scoring answerer.

This is the placeholder for the task-trained answerer. The data path and the
retrieval recall gate already live in ``answerers.modernbert_data`` /
``answerers.passages`` / ``answerers.minilm``; the cross-encoder itself is built
on IDUN (M3). Until a checkpoint exists, selecting ``MEDAPP_ANSWERER=modernbert``
raises a clear error rather than silently guessing.

Planned shape (see docs/adr/0003 and docs/sol-assestment.md):

* retrieve top-k passages with MiniLM (``modernbert_data``);
* score each ``[CLS] question [SEP] passage [SEP]`` with a ModernBERT-base
  cross-encoder carrying a 3-way SUPPORT/REFUTE/NOT_MENTIONED head, a token
  start/end head, and an expected-tIoU regression head;
* decode ``yes`` when ``p > 0.4 / (0.8 + 1.2 q)`` and return the predicted
  token span mapped back to word timestamps.
"""

from typing import Dict, List, Optional, Sequence

from .base import Answer

MODEL_CHECKPOINT = None  # set by M3 once trained


class ModernBertAnswerer:
    """Placeholder; replaced by the trained cross-encoder."""

    name = "modernbert"
    supports_info = True

    def warm_up(self) -> None:  # pragma: no cover - until M3
        if MODEL_CHECKPOINT is None:
            raise RuntimeError(
                "ModernBERT checkpoint not built yet (M3); "
                "keep MEDAPP_ANSWERER=legacy."
            )

    def answer_all(
        self,
        questions: Sequence[str],
        transcript: Dict,
        *,
        deadline: Optional[float] = None,
        return_info: bool = False,
    ) -> List[Answer]:
        raise NotImplementedError(
            "ModernBERT answerer is not trained yet (M3); "
            "keep MEDAPP_ANSWERER=legacy."
        )


def build_answerer() -> ModernBertAnswerer:
    return ModernBertAnswerer()
