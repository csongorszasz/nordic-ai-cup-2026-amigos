"""MiniLM passage retrieval for the ModernBERT answerer.

A small sentence encoder (``all-MiniLM-L6-v2``) scores every passage against the
raw question; the top-k go on to the cross-encoder. Implemented directly on
``transformers`` (mean pooling + L2 normalisation) so the serving path does not
need ``sentence-transformers``.

Weights are loaded lazily and can be pre-downloaded for offline use; see
``idun/setup_env.sh``. This model is deliberately tiny (~22M params) so it can
share the 4 GB GPU with the ASR and cross-encoder models.
"""

import logging
import os
from typing import List, Optional, Sequence, Tuple

from .passages import Passage

logger = logging.getLogger(__name__)

# multi-qa-* is trained for question->passage retrieval and beat the general
# all-MiniLM models on the recall gate (top-3 supports 0.897 vs 0.877 at 48/24).
MODEL_NAME = os.environ.get(
    "MEDAPP_MINILM", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)
MAX_LENGTH = int(os.environ.get("MEDAPP_MINILM_MAX_LENGTH", "256"))
BATCH_SIZE = int(os.environ.get("MEDAPP_MINILM_BATCH_SIZE", "64"))
DEVICE = os.environ.get("MEDAPP_MINILM_DEVICE", "auto").lower()


class MiniLMRetriever:
    """Lazy MiniLM encoder with cosine ranking."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or MODEL_NAME
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading MiniLM retriever %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        if DEVICE in ("cuda", "cpu"):
            self._device = DEVICE
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("MiniLM device=cuda but CUDA unavailable; using CPU.")
            self._device = "cpu"
        self._model.to(self._device)
        self._model.eval()

    def warm_up(self) -> None:
        self._load()

    def encode(self, texts: Sequence[str]):
        """L2-normalised embeddings as a float32 numpy array."""
        if not texts:
            import numpy as np

            return np.zeros((0, 384), dtype="float32")

        import numpy as np
        import torch
        import torch.nn.functional as functional

        self._load()
        vectors: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                chunk = list(texts[start:start + BATCH_SIZE])
                encoded = self._tokenizer(
                    chunk,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                output = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)

    def rank(
        self,
        question: str,
        passages: Sequence[Passage],
        top_k: Optional[int] = None,
        passage_embeddings=None,
    ) -> List[Tuple[Passage, float]]:
        """Passages sorted by cosine similarity to the question."""
        if not passages:
            return []

        import numpy as np

        if passage_embeddings is None:
            passage_embeddings = self.encode([p.text for p in passages])
        question_embedding = self.encode([question])[0]
        scores = passage_embeddings @ question_embedding

        order = np.argsort(-scores)
        if top_k is not None:
            order = order[:top_k]
        return [(passages[int(i)], float(scores[int(i)])) for i in order]
