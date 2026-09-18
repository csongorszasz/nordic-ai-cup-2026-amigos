"""Delta boundary refiner model: word-level adjustments to the LLM quote.

The refiner predicts how far the annotated evidence extends beyond the quoted
span on each side (``left`` = words to add before the anchor, ``right`` = words
to add after it). Predicting the *edit* rather than absolute boundaries makes
"keep the LLM span" the zero-output default, so a correct quote is never moved
to a different occurrence — the failure mode of the token-level variant.

Heavy torch imports are module-level here because only the training/eval scripts
and the lazy refiner wrapper import this module.
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get(
    "MEDAPP_REFINER_MODEL",
    os.environ.get("MEDAPP_MB_MODEL", "answerdotai/ModernBERT-base"),
)
MAX_LENGTH = int(os.environ.get("MEDAPP_REFINER_MAX_LENGTH", "256"))
MAX_DELTA = int(os.environ.get("MEDAPP_REFINER_MAX_DELTA", "12"))
DROPOUT = float(os.environ.get("MEDAPP_REFINER_DROPOUT", "0.1"))


class DeltaRefiner(nn.Module):
    """CLS-pooled encoder with two word-adjustment regression heads."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        dropout: float = DROPOUT,
        max_delta: int = MAX_DELTA,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.left = nn.Linear(hidden, 1)
        self.right = nn.Linear(hidden, 1)
        self.max_delta = max_delta

    def forward(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return {
            "left": self.left(pooled).squeeze(-1),
            "right": self.right(pooled).squeeze(-1),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)


def load_delta_refiner(
    checkpoint: Optional[str], device: str = "auto"
) -> Tuple[DeltaRefiner, str]:
    """Load a trained delta refiner from ``checkpoint`` (state dict path)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeltaRefiner()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def compute_delta_loss(
    outputs: Dict[str, torch.Tensor],
    left_targets: torch.Tensor,
    right_targets: torch.Tensor,
    max_delta: int = MAX_DELTA,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Huber loss on the clamped word adjustments."""
    left = left_targets.clamp(-max_delta, max_delta)
    right = right_targets.clamp(-max_delta, max_delta)
    left_loss = functional.smooth_l1_loss(outputs["left"], left)
    right_loss = functional.smooth_l1_loss(outputs["right"], right)
    total = left_loss + right_loss
    return total, {
        "left": float(left_loss.detach()),
        "right": float(right_loss.detach()),
    }
