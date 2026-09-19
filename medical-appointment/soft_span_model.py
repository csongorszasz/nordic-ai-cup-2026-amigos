"""Soft-target span tagger: ModernBERT token classifier trained on a soft
per-word evidence label derived from multiple annotators (official gold and the
independent agent draft), then decoded to one span and evaluated out-of-fold
against the recorded incumbent.

    python soft_span_model.py --folds 5 --epochs 4

Development-only; decisions stay fixed (the 26B decides), only the evidence span
is replaced where the tagger is confident. Never used by ``/predict``.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from answerers import modernbert_data as data
from answerers.modernbert_model import MODEL_NAME
from answerers.passages import overlap_word_range
from answerers.align import align_quote_matches

PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def word_soft_labels(words, spans, mode="union"):
    """Per-word evidence target: ``union`` (any annotator) or ``soft`` (vote share)."""
    n = len(words)
    if not spans:
        return [0.0] * n
    counts = [0.0] * n
    for span in spans:
        rng = overlap_word_range(words, span[0], span[1])
        if rng is None:
            continue
        for index in range(rng[0], rng[1] + 1):
            counts[index] += 1.0
    if mode == "soft":
        return [value / len(spans) for value in counts]
    return [1.0 if value > 0 else 0.0 for value in counts]


def build_examples(base, transcripts, evidence, drafts, target="union"):
    examples = []
    for qid, record in base.items():
        if record.get("label", 1) != 1 or not record.get("answer"):
            continue
        tid = record["transcript_id"]
        words = transcripts[tid]["words"]
        spans = []
        gold = record.get("gold") or (
            [float(evidence[qid]["start"]), float(evidence[qid]["end"])]
            if evidence.get(qid, {}).get("start") else None
        )
        if gold:
            spans.append(gold)
        agent_quote = drafts.get(qid, {}).get("quote")
        if agent_quote:
            matches = align_quote_matches(words, agent_quote)
            if matches:
                spans.append([matches[0][0], matches[0][1]])
        if not spans:
            continue
        examples.append(
            {
                "question_id": qid,
                "transcript_id": tid,
                "words": ["".join(w["word"].split()) or " " for w in words],
                "soft": word_soft_labels(words, spans, target),
                "base_span": record.get("span"),
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=PROJECT_ROOT / "results" / "llm_p_base_L1_questions.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--target", choices=("union", "soft"), default="union",
                        help="Per-word target from multiple annotators.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "soft_span_questions.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    base = {row["question_id"]: row for row in json.loads(args.base.read_text())}
    evidence = data.load_evidence()
    drafts = {r["question_id"]: r for r in json.loads(
        (PROJECT_ROOT / "annotations" / "drafts" / "ALL.json").read_text())}
    tids = sorted({row["transcript_id"] for row in base.values()})
    transcripts = {tid: data.load_transcript(tid) for tid in tids}
    examples = build_examples(base, transcripts, evidence, drafts, args.target)
    logger.info("soft examples: %d", len(examples))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class SoftDataset(Dataset):
        def __init__(self, items):
            self.items = []
            for example in items:
                encoding = tokenizer(
                    example["words"], is_split_into_words=True,
                    truncation=True, max_length=args.max_length,
                )
                labels = [
                    -100.0 if wid is None else float(example["soft"][wid])
                    for wid in encoding.word_ids()
                ]
                self.items.append(
                    {"input_ids": encoding["input_ids"],
                     "attention_mask": encoding["attention_mask"],
                     "labels": labels, "example": example}
                )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def collate(batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids, attention, labels = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100.0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention),
            "labels": torch.tensor(labels),
            "examples": [item["example"] for item in batch],
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    folds = data.grouped_folds(tids, args.folds, args.seed)

    def train_model(train_items):
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()), lr=args.lr
        )
        positives = sum(1 for e in train_items for value in e["soft"] if value > 0.5)
        negatives = sum(1 for e in train_items for value in e["soft"] if value <= 0.5)
        pos_weight = torch.tensor([negatives / max(1, positives)], device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        loader = DataLoader(SoftDataset(train_items), batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate)
        model.train()
        for epoch in range(args.epochs):
            total = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                hidden = model(input_ids=input_ids, attention_mask=attention).last_hidden_state
                logits = head(hidden).squeeze(-1)
                mask = labels != -100
                loss = loss_fn(logits[mask], labels[mask]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            logger.info("epoch %d loss %.4f", epoch + 1, total / max(1, len(loader)))
        model.eval()
        return model, head

    @torch.no_grad()
    def predict(model, head, example):
        encoding = tokenizer(
            example["words"], is_split_into_words=True,
            truncation=True, max_length=args.max_length, return_tensors="pt",
        )
        word_ids = encoding.word_ids()
        logits = head(model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        ).last_hidden_state).squeeze(-1)[0].cpu()
        probs = torch.sigmoid(logits).tolist()
        buckets = defaultdict(list)
        for token_index, wid in enumerate(word_ids):
            if wid is not None:
                buckets[wid].append(probs[token_index])
        word_probs = [sum(buckets[i]) / len(buckets[i]) if buckets[i] else 0.0
                      for i in range(len(example["words"]))]
        mean_prob = sum(word_probs) / len(word_probs) if word_probs else 0.0
        # Maximum-subarray on (p - mean): the contiguous run with the strongest
        # evidence excess, so a span is always produced despite class imbalance.
        best, best_sum = None, 0.0
        current, current_sum = None, 0.0
        for index, probability in enumerate(word_probs):
            excess = probability - mean_prob
            if current is None or current_sum + excess < excess:
                current, current_sum = [index], excess
            else:
                current.append(index)
                current_sum += excess
            if current_sum > best_sum:
                best, best_sum = list(current), current_sum
        if not best:
            return example["base_span"], 0.0
        words = transcripts[example["transcript_id"]]["words"]
        return [words[best[0]]["start"], words[best[-1]]["end"]], best_sum

    changed = 0
    for fold_index, held_out in enumerate(folds):
        held = set(held_out)
        train_items = [e for e in examples if e["transcript_id"] not in held]
        test_items = [e for e in examples if e["transcript_id"] in held]
        logger.info("fold %d train=%d test=%d", fold_index, len(train_items), len(test_items))
        model, head = train_model(train_items)
        for example in test_items:
            span, confidence = predict(model, head, example)
            record = base[example["question_id"]]
            if span is not None and confidence > 0:
                record["span"] = list(span)
                changed += 1

    args.output.write_text(json.dumps(list(base.values()), indent=2))
    print(f"soft-span OOF changed={changed} wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
