"""Train the ModernBERT candidate scorer on the supplied conversations.

Conversation-grouped: ``--folds`` GroupKFold-by-conversation folds, or
``--loco`` for leave-one-conversation-out. Every fold's held-out conversations
are scored with the *serving* code path (``predict_transcript``) and written to
``oof_modernbert.json``, so the out-of-fold predictions can be calibrated and
compared with the legacy OOF (``results/oof_T030.json``) before anything is
flipped.

Designed for IDUN (fp16-capable GPU required by the job script):

    sbatch idun/job_train_modernbert.slurm --folds 5

Never trains on validation/evaluation data: only the 39 supplied conversations
and ``annotations/evidence.csv`` are read.
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
from answerers import modernbert_model as model_module  # noqa: E402
from answerers.minilm import MiniLMRetriever  # noqa: E402
from answerers.modernbert import predict_transcript  # noqa: E402
from utils import temporal_iou  # noqa: E402


def grouped_folds(tids, n_folds: int, seed: int = 0):
    """Conversation-grouped folds: no conversation is split across folds."""
    shuffled = list(tids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


def native_score(records):
    correct = 0
    tious = []
    for record in records:
        correct += int(bool(record["answer"]) == bool(record["label"]))
        if record["label"] == 1 and record["gold"] is not None:
            span = tuple(record["span"]) if record["span"] else None
            tious.append(temporal_iou(tuple(record["gold"]), span))
    accuracy = correct / len(records) if records else 0.0
    mean_tiou = sum(tious) / len(tious) if tious else 0.0
    return 0.4 * accuracy + 0.6 * mean_tiou, accuracy, mean_tiou


def train_fold(examples, tokenizer, args, device):
    model = model_module.ModernBertScorer()
    model.to(device)
    model.train()

    dataset = model_module.WordSpanDataset(examples, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=model_module.Collator(tokenizer),
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
                    loss, parts = model_module.compute_loss(
                        outputs, tensors, args.span_weight, args.tiou_weight
                    )
            else:
                outputs = model(
                    input_ids=tensors["input_ids"],
                    attention_mask=tensors["attention_mask"],
                )
                loss, parts = model_module.compute_loss(
                    outputs, tensors, args.span_weight, args.tiou_weight
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
            f"(cls {running['cls'] / steps:.4f} span {running['span'] / steps:.4f} "
            f"tiou {running['tiou'] / steps:.4f})"
        )
    return model


def evaluate_conversations(model, tokenizer, retriever, args, device, tids, words_cache, rows_by_tid):
    model.eval()
    records = []
    for tid in tids:
        words = words_cache[tid]
        rows = rows_by_tid[tid]
        questions = [row["question"] for row in rows]
        results = predict_transcript(
            model, tokenizer, retriever, model_module, device, words, questions
        )
        for row, (answer, span, info) in zip(rows, results):
            records.append(
                {
                    "question_id": row["question_id"],
                    "transcript_id": tid,
                    "question_type": row["question_type"],
                    "label": int(row["label"]),
                    "p": info.get("p"),
                    "answer": bool(answer),
                    "span": list(span) if span is not None else None,
                    "gold": (
                        [float(row["evidence_start"]), float(row["evidence_end"])]
                        if row["question_type"] == "positive"
                        else None
                    ),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models/modernbert")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--span-weight", type=float, default=1.0)
    parser.add_argument("--tiou-weight", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loco", action="store_true",
                        help="Leave-one-conversation-out (39 folds; slow).")
    parser.add_argument("--top-k", type=int, default=data.TOP_K)
    parser.add_argument("--cross-negatives", type=int, default=2)
    parser.add_argument("--window", type=int, default=data.WINDOW_WORDS)
    parser.add_argument("--stride", type=int, default=data.STRIDE_WORDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device; training on CPU will be very slow.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("building candidate examples ...")
    examples = data.build_examples(
        window=args.window, stride=args.stride, top_k=args.top_k,
        n_cross_negatives=args.cross_negatives, limit=args.limit, seed=args.seed,
    )
    print(f"  {len(examples)} examples")

    rows = data.load_rows()
    if args.limit:
        rows = rows[: args.limit]
    rows_by_tid: dict = defaultdict(list)
    for row in rows:
        rows_by_tid[row["transcript_id"]].append(row)
    words_cache = {tid: data.load_words(tid) for tid in rows_by_tid}

    tids = sorted(rows_by_tid)
    if args.loco:
        folds = [[tid] for tid in tids]
    else:
        folds = grouped_folds(tids, args.folds, args.seed)
    print(f"  {len(tids)} conversations -> {len(folds)} folds")

    tokenizer = model_module.AutoTokenizer.from_pretrained(model_module.MODEL_NAME)
    retriever = MiniLMRetriever()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_oof = []
    started = time.time()
    for fold_index, held_out in enumerate(folds):
        print(f"\nfold {fold_index + 1}/{len(folds)}: val={held_out}")
        held = set(held_out)
        train_examples = [e for e in examples if e["transcript_id"] not in held]
        model = train_fold(train_examples, tokenizer, args, device)
        checkpoint = output_dir / f"fold_{fold_index}.pt"
        model.save(str(checkpoint))

        oof = evaluate_conversations(
            model, tokenizer, retriever, args, device,
            held_out, words_cache, rows_by_tid,
        )
        score, accuracy, mean_tiou = native_score(oof)
        print(
            f"  val score {score:.3f}  acc {accuracy:.3f}  mIoU {mean_tiou:.3f} "
            f"({len(oof)} questions)"
        )
        all_oof.extend(oof)

    score, accuracy, mean_tiou = native_score(all_oof)
    print(
        f"\nOOF native: score {score:.3f}  acc {accuracy:.3f}  mIoU {mean_tiou:.3f}"
    )
    oof_path = output_dir / "oof_modernbert.json"
    oof_path.write_text(json.dumps(all_oof, indent=2))
    summary = {
        "tag": "T-?",
        "epochs": args.epochs,
        "folds": len(folds),
        "loco": args.loco,
        "window": args.window,
        "stride": args.stride,
        "top_k": args.top_k,
        "score": round(score, 4),
        "accuracy": round(accuracy, 4),
        "mean_tiou": round(mean_tiou, 4),
        "conversations": len(tids),
        "examples": len(examples),
        "elapsed_s": round(time.time() - started, 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOOF -> {oof_path}\nsummary -> {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
