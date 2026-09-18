"""The ModernBERT candidate scorer: model, dataset, losses, decoding.

Imported lazily by ``answerers.modernbert`` only when a checkpoint is actually
used, so the torch-free environments (tests, ``dev_eval --answer nli``) never
load it.

One pair per candidate passage::

    [CLS] question [SEP] passage [SEP]

with three heads on top of the encoder:

* 3-way classification SUPPORT / REFUTE / NOT_MENTIONED (CLS pooled);
* token start/end logits (SQuAD-style) for the evidence span;
* expected-tIoU regression (sigmoid on the CLS pooled vector).
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

from .passages import Passage
from .span_utils import (
    LABEL_NAMES,
    LABEL_NOT_MENTIONED,
    LABEL_REFUTE,
    LABEL_SUPPORT,
    char_span_to_words,
    target_token_span,
    token_span_to_char,
    word_char_spans,
)

MODEL_NAME = os.environ.get("MEDAPP_MB_MODEL", "answerdotai/ModernBERT-base")
MAX_LENGTH = int(os.environ.get("MEDAPP_MB_MAX_LENGTH", "512"))
DROPOUT = float(os.environ.get("MEDAPP_MB_DROPOUT", "0.1"))

LABEL_TO_ID = {"support": LABEL_SUPPORT, "refute": LABEL_REFUTE,
               "not_mentioned": LABEL_NOT_MENTIONED}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class ModernBertScorer(nn.Module):
    """ModernBERT encoder with classification, span and tIoU heads."""

    def __init__(self, model_name: str = MODEL_NAME, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, len(LABEL_NAMES))
        self.span_start = nn.Linear(hidden, 1)
        self.span_end = nn.Linear(hidden, 1)
        self.tiou = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return {
            "logits": self.classifier(pooled),
            "start": self.span_start(hidden).squeeze(-1),
            "end": self.span_end(hidden).squeeze(-1),
            "tiou": torch.sigmoid(self.tiou(pooled)).squeeze(-1),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)


def load_scorer(checkpoint: Optional[str], device: str = "auto") -> Tuple[ModernBertScorer, str]:
    """Load a trained scorer from ``checkpoint`` (state dict path)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernBertScorer()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def encode_pair(tokenizer, question: str, passage_text: str, max_length: int = MAX_LENGTH):
    """Tokenize one pair, keeping offsets and sequence ids for span mapping."""
    encoding = tokenizer(
        question,
        passage_text,
        truncation="only_second",
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    return encoding


def collate_pairs(tokenizer, pairs: Sequence[Tuple[str, str]], max_length: int = MAX_LENGTH):
    """Dynamic-pad a list of ``(question, passage)`` pairs.

    Returns ``(batch_tensors, offsets, sequence_ids)`` where the latter two stay
    as Python lists so the caller can map token spans back to text.
    """
    encodings = [encode_pair(tokenizer, q, p, max_length) for q, p in pairs]
    offsets = [enc["offset_mapping"] for enc in encodings]
    sequence_ids = [list(enc.sequence_ids()) for enc in encodings]
    batch = tokenizer.pad(
        [
            {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            for enc in encodings
        ],
        padding=True,
        return_tensors="pt",
    )
    return batch, offsets, sequence_ids


class WordSpanDataset(torch.utils.data.Dataset):
    """Tokenized training features from ``modernbert_data.build_examples``."""

    def __init__(self, examples: Sequence[Dict], tokenizer, max_length: int = MAX_LENGTH):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index]
        text, word_spans = word_char_spans(example.get("passage_words") or [])
        encoding = encode_pair(self.tokenizer, example["question"], text, self.max_length)
        sequence_ids = list(encoding.sequence_ids())

        start = end = -100
        span_words = example.get("span_words")
        if span_words and word_spans:
            first = max(0, min(span_words[0], len(word_spans) - 1))
            last = max(0, min(span_words[1], len(word_spans) - 1))
            char_start = word_spans[first][0]
            char_end = word_spans[last][1]
            span = target_token_span(
                encoding["offset_mapping"], sequence_ids, char_start, char_end
            )
            if span is not None:
                start, end = span

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": LABEL_TO_ID[example["label"]],
            "start_positions": start,
            "end_positions": end,
            "tiou_target": float(example.get("target_tiou", 0.0)),
            "meta": example,
        }


class Collator:
    """Pad a batch of dataset items into tensors, keeping metadata aside."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Dict]) -> Dict:
        batch = self.tokenizer.pad(
            [
                {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                for f in features
            ],
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        batch["start_positions"] = torch.tensor(
            [f["start_positions"] for f in features], dtype=torch.long
        )
        batch["end_positions"] = torch.tensor(
            [f["end_positions"] for f in features], dtype=torch.long
        )
        batch["tiou_target"] = torch.tensor(
            [f["tiou_target"] for f in features], dtype=torch.float
        )
        batch["meta"] = [f["meta"] for f in features]
        return batch


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    span_weight: float = 1.0,
    tiou_weight: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """3-way CE + span CE (support/refute only) + expected-tIoU MSE.

    ``class_weights`` counteracts the ~5 % SUPPORT rate so the decision head is
    not biased to NOT_MENTIONED.
    """
    cls_loss = functional.cross_entropy(
        outputs["logits"], batch["labels"], weight=class_weights
    )

    span_mask = batch["start_positions"] >= 0
    if span_mask.any():
        start_loss = functional.cross_entropy(
            outputs["start"][span_mask], batch["start_positions"][span_mask]
        )
        end_loss = functional.cross_entropy(
            outputs["end"][span_mask], batch["end_positions"][span_mask]
        )
        span_loss = start_loss + end_loss
    else:
        span_loss = torch.zeros((), device=cls_loss.device)

    tiou_loss = functional.mse_loss(outputs["tiou"], batch["tiou_target"])
    total = cls_loss + span_weight * span_loss + tiou_weight * tiou_loss
    return total, {
        "cls": float(cls_loss.detach()),
        "span": float(span_loss.detach()),
        "tiou": float(tiou_loss.detach()),
    }


def _best_passage_token(logits: torch.Tensor, sequence_ids: Sequence) -> int:
    """Argmax over passage tokens only."""
    masked = logits.clone()
    for i, seq in enumerate(sequence_ids):
        if seq != 1:
            masked[i] = float("-inf")
    return int(torch.argmax(masked).item())


def predicted_span_seconds(
    passage: Passage,
    words: Sequence[Dict],
    offsets: Sequence,
    sequence_ids: Sequence,
    start_index: int,
    end_index: int,
) -> Optional[Tuple[float, float]]:
    """Map a predicted token span back to an audio ``[start, end]``."""
    passage_words = [
        words[k]["word"].strip()
        for k in range(passage.first_word, passage.last_word + 1)
    ]
    _, word_spans = word_char_spans(passage_words)
    char = token_span_to_char(offsets, sequence_ids, start_index, end_index)
    if char is None:
        return None
    relative = char_span_to_words(word_spans, char[0], char[1])
    if relative is None:
        return None
    first = passage.first_word + relative[0]
    last = passage.first_word + relative[1]
    return float(words[first]["start"]), float(words[last]["end"])


@torch.no_grad()
def score_pairs(model, tokenizer, pairs, device, max_length: int = MAX_LENGTH):
    """Score a list of ``(question, passage)`` pairs.

    Returns a list of dicts per pair with ``probs``, ``label``, ``p_support``,
    ``expected_tiou``, ``start_index``, ``end_index``, ``offsets``,
    ``sequence_ids``.
    """
    batch, offsets, sequence_ids = collate_pairs(tokenizer, pairs, max_length)
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    probs = torch.softmax(outputs["logits"], dim=-1).cpu()
    tiou = outputs["tiou"].cpu()
    starts = outputs["start"].cpu()
    ends = outputs["end"].cpu()

    results = []
    for i in range(len(pairs)):
        row = probs[i]
        label = int(torch.argmax(row).item())
        start_index = _best_passage_token(starts[i], sequence_ids[i])
        end_index = _best_passage_token(ends[i], sequence_ids[i])
        results.append(
            {
                "probs": row.tolist(),
                "label": label,
                "p_support": float(row[LABEL_SUPPORT]),
                "expected_tiou": float(tiou[i]),
                "start_index": start_index,
                "end_index": end_index,
                "offsets": offsets[i],
                "sequence_ids": sequence_ids[i],
            }
        )
    return results
