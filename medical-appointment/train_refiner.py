"""Train the post-hoc boundary refiner on the 39 supplied conversations.

The refiner is anchored on the E4B probe's own occurrence:
``answerers.refiner_data.build_refiner_examples`` builds a word window around
the LLM quote; the model reads the marked window and predicts word adjustments
(``left``/``right``) that recover the annotated extent. Predicting the edit
makes "keep the LLM span" the zero default, and evaluation runs the serving
decode path over conversation-grouped folds or leave-one-conversation-out.

    sbatch idun/job_train_refiner.slurm --folds 5 --bf16
    sbatch idun/job_train_refiner.slurm --loco --bf16
    sbatch idun/job_train_refiner.slurm --train-all --bf16

Writes ``results/refiner_<tag>.json`` (summary) and
``results/refiner_<tag>_questions.json`` (per-question records).
Never trains on validation/evaluation data: only the 39 supplied conversations,
``annotations/evidence.csv`` golds and the in-sample E4B probe records are read.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from answerers import modernbert_data as data  # noqa: E402
from answerers import refiner as refiner_module  # noqa: E402
from answerers import refiner_data  # noqa: E402
from answerers import refiner_model  # noqa: E402
from utils import temporal_iou  # noqa: E402


def grouped_folds(tids, n_folds: int, seed: int = 0):
    """Conversation-grouped folds: no conversation is split across folds."""
    shuffled = list(tids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


class DeltaDataset:
    """Marked-window encodings with left/right word-adjustment targets."""

    def __init__(self, examples, tokenizer, max_length: int = 256):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        llm_first, llm_last = example["llm_word_range"]
        gold_first, gold_last = example["gold_word_range"]
        text, _ = refiner_module.marked_window_layout(
            example["passage_words"],
            llm_first - example["first_word"],
            llm_last - example["first_word"],
        )
        encoding = self.tokenizer(
            example["question"],
            text,
            truncation="only_second",
            max_length=self.max_length,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "left_target": float(llm_first - gold_first),
            "right_target": float(gold_last - llm_last),
            "meta": example,
        }


class DeltaCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [
                {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
                for f in features
            ],
            padding=True,
            return_tensors="pt",
        )
        batch["left_target"] = torch.tensor(
            [f["left_target"] for f in features], dtype=torch.float
        )
        batch["right_target"] = torch.tensor(
            [f["right_target"] for f in features], dtype=torch.float
        )
        batch["meta"] = [f["meta"] for f in features]
        return batch


def train_fold(examples, tokenizer, args, device):
    model = refiner_model.DeltaRefiner(max_delta=args.max_delta)
    model.to(device)
    model.train()

    loader = DataLoader(
        DeltaDataset(examples, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DeltaCollator(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = args.bf16 and device == "cuda"

    for epoch in range(args.epochs):
        running = defaultdict(float)
        for batch in loader:
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            optimizer.zero_grad()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=tensors["input_ids"],
                        attention_mask=tensors["attention_mask"],
                    )
                    loss, parts = refiner_model.compute_delta_loss(
                        outputs, tensors["left_target"], tensors["right_target"],
                        args.max_delta,
                    )
            else:
                outputs = model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                )
                loss, parts = refiner_model.compute_delta_loss(
                    outputs, tensors["left_target"], tensors["right_target"],
                    args.max_delta,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running["loss"] += float(loss.detach())
            for key, value in parts.items():
                running[key] += value
        steps = max(1, len(loader))
        print(
            f"    epoch {epoch + 1}/{args.epochs} "
            f"loss {running['loss'] / steps:.4f} "
            f"(left {running['left'] / steps:.4f} right {running['right'] / steps:.4f})"
        )
    return model


def evaluate_examples(model, tokenizer, args, device, examples):
    """Refine every example with the serving decode path; return records."""
    model.eval()
    refiner = refiner_module.BoundaryRefiner(
        checkpoint="in-memory", window=args.window, max_length=args.max_length
    )
    refiner._model = model
    refiner._tokenizer = tokenizer
    refiner._device = device
    refiner._module = refiner_model

    records = []
    words_cache = {}
    for example in examples:
        tid = example["transcript_id"]
        if tid not in words_cache:
            words_cache[tid] = data.load_words(tid)
        words = words_cache[tid]
        llm_first, llm_last = example["llm_word_range"]
        span = refiner.refine(example["question"], words, llm_first, llm_last)

        gold = tuple(example["span"])
        llm_span = tuple(example["llm_span"])
        records.append(
            {
                "question_id": example["question_id"],
                "transcript_id": example["transcript_id"],
                "gold": list(gold),
                "llm_span": list(llm_span),
                "span": list(span) if span is not None else None,
                "tiou": round(temporal_iou(gold, span), 4),
                "llm_tiou": round(temporal_iou(gold, llm_span), 4),
            }
        )
    return records


def bucket_counts(values):
    buckets = {">=0.8": 0, "0.5-0.8": 0, "0.3-0.5": 0, "0.1-0.3": 0, "<0.1": 0}
    for value in values:
        if value >= 0.8:
            buckets[">=0.8"] += 1
        elif value >= 0.5:
            buckets["0.5-0.8"] += 1
        elif value >= 0.3:
            buckets["0.3-0.5"] += 1
        elif value >= 0.1:
            buckets["0.1-0.3"] += 1
        else:
            buckets["<0.1"] += 1
    return buckets


def summarize(records, llm_questions_path):
    """Refiner vs LLM mIoU plus the composed end-to-end in-sample score.

    Uncovered positives (gold outside the LLM window or the LLM answered no)
    fall back to the LLM span, mirroring what the service would do.
    """
    llm = refiner_data.load_llm_records(llm_questions_path)
    rows = [row for row in data.load_rows() if row["question_type"] == "positive"]
    covered = {record["question_id"]: record for record in records}

    refiner_tious = [record["tiou"] for record in records]
    baseline_tious = [record["llm_tiou"] for record in records]

    composed = []
    llm_all = []
    for row in rows:
        gold = (float(row["evidence_start"]), float(row["evidence_end"]))
        entry = llm.get(row["question_id"], {})
        llm_span = (
            tuple(entry["span"]) if entry.get("answer") and entry.get("span") else None
        )
        llm_all.append(temporal_iou(gold, llm_span))
        if row["question_id"] in covered:
            record = covered[row["question_id"]]
            span = tuple(record["span"]) if record["span"] else None
            composed.append(temporal_iou(gold, span))
        else:
            composed.append(temporal_iou(gold, llm_span))

    predictions = [int(bool(entry.get("answer"))) for entry in llm.values()]
    labels = [int(bool(entry.get("label"))) for entry in llm.values()]
    accuracy = (
        sum(p == y for p, y in zip(predictions, labels)) / len(labels)
        if labels else 0.0
    )
    composed_miou = sum(composed) / len(composed) if composed else 0.0
    llm_miou_all = sum(llm_all) / len(llm_all) if llm_all else 0.0
    n = len(records) or 1

    return {
        "examples": len(records),
        "covered_positives": len(records),
        "refiner_miou": round(sum(refiner_tious) / n, 4),
        "llm_miou_on_covered": round(sum(baseline_tious) / n, 4),
        "refiner_buckets": bucket_counts(refiner_tious),
        "llm_buckets": bucket_counts(baseline_tious),
        "composed_miou": round(composed_miou, 4),
        "llm_miou_all_positives": round(llm_miou_all, 4),
        "llm_accuracy": round(accuracy, 4),
        "llm_score_baseline": round(0.4 * accuracy + 0.6 * llm_miou_all, 4),
        "composed_score": round(0.4 * accuracy + 0.6 * composed_miou, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-questions",
                        default=refiner_data.DEFAULT_LLM_QUESTIONS)
    parser.add_argument("--output-dir", default="models/refiner")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tag", default="e4b")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=refiner_model.MAX_LENGTH)
    parser.add_argument("--max-delta", type=int, default=refiner_model.MAX_DELTA)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loco", action="store_true",
                        help="Leave-one-conversation-out (39 folds; slow).")
    parser.add_argument("--train-all", action="store_true",
                        help="Train one model on every example (serving); "
                             "writes final.pt and skips OOF.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device; training on CPU will be very slow.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("building anchored refiner examples ...")
    examples, skipped = refiner_data.build_refiner_examples(
        args.llm_questions, window=args.window, limit=args.limit
    )
    print(f"  {len(examples)} examples; skipped: {skipped}")
    if not examples:
        print("no examples — is the E4B probe records file present?")
        return 1

    tokenizer = refiner_model.AutoTokenizer.from_pretrained(refiner_model.MODEL_NAME)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_all:
        print(f"training on all {len(examples)} examples (serving refiner)")
        model = train_fold(examples, tokenizer, args, device)
        model.save(str(output_dir / "final.pt"))
        print(f"final -> {output_dir / 'final.pt'}")
        return 0

    examples_by_tid: dict = defaultdict(list)
    for example in examples:
        examples_by_tid[example["transcript_id"]].append(example)
    tids = sorted(examples_by_tid)
    if args.loco:
        folds = [[tid] for tid in tids]
    else:
        folds = grouped_folds(tids, args.folds, args.seed)
    print(f"  {len(tids)} conversations -> {len(folds)} folds")

    all_records = []
    started = time.time()
    for fold_index, held_out in enumerate(folds):
        held = set(held_out)
        train_examples = [
            example for example in examples if example["transcript_id"] not in held
        ]
        eval_examples = [
            example for tid in held_out for example in examples_by_tid.get(tid, [])
        ]
        if not eval_examples:
            continue
        print(f"\nfold {fold_index + 1}/{len(folds)}: val={held_out} "
              f"({len(train_examples)} train / {len(eval_examples)} eval)")
        model = train_fold(train_examples, tokenizer, args, device)
        records = evaluate_examples(model, tokenizer, args, device, eval_examples)
        fold_tiou = sum(record["tiou"] for record in records) / len(records)
        fold_llm = sum(record["llm_tiou"] for record in records) / len(records)
        print(f"  refiner mIoU {fold_tiou:.3f} vs LLM {fold_llm:.3f} "
              f"({len(records)} questions)")
        all_records.extend(records)

    summary = summarize(all_records, args.llm_questions)
    summary.update(
        {
            "tag": args.tag,
            "model": refiner_model.MODEL_NAME,
            "window": args.window,
            "max_delta": args.max_delta,
            "epochs": args.epochs,
            "folds": len(folds),
            "loco": args.loco,
            "conversations": len(tids),
            "skipped": skipped,
            "elapsed_s": round(time.time() - started, 1),
        }
    )
    print(
        f"\nOOF refiner mIoU {summary['refiner_miou']:.3f} "
        f"(LLM on same {summary['llm_miou_on_covered']:.3f})  "
        f"composed mIoU {summary['composed_miou']:.3f}  "
        f"composed score {summary['composed_score']:.3f}"
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"refiner_{args.tag}.json").write_text(
        json.dumps(summary, indent=2)
    )
    (results_dir / f"refiner_{args.tag}_questions.json").write_text(
        json.dumps(all_records, indent=2)
    )
    print(f"\nsummary -> {results_dir / f'refiner_{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
