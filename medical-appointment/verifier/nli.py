"""NLI cross-encoder: score how strongly a window entails a proposition.

Wraps a DeBERTa-v3 MNLI model. The model never sees timestamps — it selects
*text*; the caller maps the chosen text back to a span. Loaded lazily and
warmed at import time in the serving path.

Environment overrides:

* ``NLI_MODEL``       default ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``
* ``NLI_BATCH_SIZE``  default ``16``
* ``NLI_MAX_LENGTH``  default ``512``
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get(
    "NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)
BATCH_SIZE = int(os.environ.get("NLI_BATCH_SIZE", "16"))
MAX_LENGTH = int(os.environ.get("NLI_MAX_LENGTH", "512"))
# "auto" picks CUDA when available; force "cpu" to keep VRAM for the ASR model.
NLI_DEVICE = os.environ.get("NLI_DEVICE", "auto").lower()
NLI_HALF = os.environ.get("NLI_HALF", "0") == "1"

_model = None
_tokenizer = None
_entail_index = 0
_device = "cpu"


def _find_entail_index(config) -> int:
    """Locate the entailment label — models differ in label order."""
    id2label = getattr(config, "id2label", {}) or {}
    for index, label in id2label.items():
        if "entail" in str(label).lower():
            try:
                return int(index)
            except (TypeError, ValueError):
                continue
    label2id = getattr(config, "label2id", {}) or {}
    for label, index in label2id.items():
        if "entail" in str(label).lower():
            try:
                return int(index)
            except (TypeError, ValueError):
                continue
    logger.warning("No entailment label found; assuming index 0.")
    return 0


def get_model():
    global _model, _tokenizer, _entail_index, _device
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading NLI model %s", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        if NLI_DEVICE in ("cuda", "cpu"):
            _device = NLI_DEVICE
        else:
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        if _device == "cuda" and not torch.cuda.is_available():
            logger.warning("NLI_DEVICE=cuda but CUDA is unavailable; using CPU.")
            _device = "cpu"
        _model.to(_device)
        if NLI_HALF and _device == "cuda":
            _model.half()
        _model.eval()
        _entail_index = _find_entail_index(_model.config)
        logger.info(
            "NLI model on %s; id2label=%s; entailment index=%d",
            _device,
            getattr(_model.config, "id2label", None),
            _entail_index,
        )
    return _model


def warm_up() -> None:
    get_model()


def entail_index() -> int:
    get_model()
    return _entail_index


def score(premises: List[str], hypothesis: str, batch_size: int = BATCH_SIZE) -> List[float]:
    """Entailment probability of ``hypothesis`` for each premise."""
    if not premises:
        return []

    import torch

    get_model()
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(premises), batch_size):
            chunk = premises[start:start + batch_size]
            encoded = _tokenizer(
                chunk,
                [hypothesis] * len(chunk),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(_device) for key, value in encoded.items()}
            logits = _model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores.extend(probabilities[:, _entail_index].cpu().tolist())
    return scores
