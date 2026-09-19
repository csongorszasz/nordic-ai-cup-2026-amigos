"""CPU feasibility/pilot for contextual word boundaries and exact span risk."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from pathlib import Path

from answerers.lora_data import split_rows
from answerers.span_lattice import local_span_lattice, marked_word_char_spans, word_token_indices
from answerers.span_utils import word_char_spans
from benchmark import paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs
from llm_probe import score_records
from train_quote_oof import validate_coverage
from utils import temporal_iou


MODEL = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
SEED = 13
EPOCHS = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-encoder", action="store_true",
                        help="Fine-tune the public encoder for a fixed three-epoch pilot.")
    parser.add_argument("--mark-anchor", action="store_true",
                        help="Show the encoder the existing quote boundaries without exposing labels.")
    parser.add_argument("--output", type=Path, default=Path("results/local_span_risk"))
    args = parser.parse_args()
    rows, requests, transcripts, _ = load_inputs(args.baseline)
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    training, validation, folds = split_rows(rows, excluded, args.fold, SEED)
    eligible_training = [row for row in training if row["answer"]]
    if not eligible_training:
        raise ValueError("No training predictions have a usable baseline word anchor.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh local-span experiment directory.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This frozen-feature feasibility/pilot is CPU-only.")
    encoder_output = Path("models/local_span_encoder")
    if args.train_encoder and encoder_output.exists():
        raise FileExistsError("Use a fresh encoder checkpoint destination.")
    import resource
    import torch
    from transformers import AutoModel, AutoTokenizer

    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if threads < 1:
        raise ValueError("Allocated CPU count must be positive.")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    candidates = {}
    selected = eligible_training if args.smoke else [
        *eligible_training, *(row for row in validation if row["answer"]),
    ]
    for row in selected:
        tid = row["transcript_id"]
        candidates[row["question_id"]] = local_span_lattice(
            transcripts[tid]["words"], row["word_range"],
            baseline_prediction(row)["span"], row["duration"],
        )
    if args.smoke:
        selected = sorted(
            selected,
            key=lambda row: (
                -(candidates[row["question_id"]]["last_word"] - candidates[row["question_id"]]["first_word"]),
                row["question_id"],
            ),
        )[:8]
        eligible_training = selected
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False,
    )
    prepared = {}
    for row in selected:
        started = time.monotonic()
        case = candidates[row["question_id"]]
        words = transcripts[row["transcript_id"]]["words"][case["first_word"]:case["last_word"] + 1]
        text, word_spans = (
            marked_word_char_spans([word["word"] for word in words], case["anchor"])
            if args.mark_anchor else word_char_spans([word["word"] for word in words])
        )
        encoding = tokenizer(
            row["question"], text, truncation=False, padding=False, return_offsets_mapping=True,
        )
        if len(encoding["input_ids"]) > 512:
            raise ValueError(f"Do not truncate local evidence context: {row['question_id']}")
        mapping = word_token_indices(encoding["offset_mapping"], encoding.sequence_ids(), word_spans)
        prepared[row["question_id"]] = {
            "inputs": {key: torch.tensor([encoding[key]], dtype=torch.long)
                       for key in ("input_ids", "attention_mask")},
            "mapping": mapping, "prepare_s": time.monotonic() - started,
        }
    encoder = AutoModel.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.float32, attn_implementation="eager", trust_remote_code=False,
    ).eval()
    encoder.requires_grad_(False)

    def encode_vectors(entry):
        hidden = encoder(**entry["inputs"]).last_hidden_state[0]
        return torch.stack([hidden[indices].mean(dim=0) for indices in entry["mapping"]])

    first = prepared[selected[0]["question_id"]]
    with torch.no_grad():
        encoder(**first["inputs"])
    features, encoding_times = {}, {}
    for row in selected:
        qid = row["question_id"]
        started = time.monotonic()
        entry = prepared[qid]
        with torch.no_grad():
            vectors = encode_vectors(entry)
        if not torch.isfinite(vectors).all():
            raise RuntimeError(f"Non-finite frozen word features for {qid}.")
        case = candidates[qid]
        features[qid] = {
            "words": vectors,
            "pairs": torch.tensor(case["pairs"], dtype=torch.long),
            "prior": torch.tensor(case["priors"], dtype=torch.float32),
        }
        encoding_times[qid] = entry["prepare_s"] + time.monotonic() - started
    hidden_size = encoder.config.hidden_size
    del vectors
    if not args.train_encoder:
        del encoder, prepared
    gc.collect()
    head = torch.nn.Linear(hidden_size, 2, bias=False)
    torch.nn.init.zeros_(head.weight)

    def logits(qid, cached=False):
        feature = features[qid]
        words = encode_vectors(prepared[qid]) if args.train_encoder and not cached else feature["words"]
        endpoint = head(words)
        pairs = feature["pairs"]
        return endpoint[pairs[:, 0], 0] + endpoint[pairs[:, 1], 1] + feature["prior"]

    with torch.no_grad():
        if any(int(logits(row["question_id"], cached=True).argmax()) != 0 for row in selected):
            raise RuntimeError("The zero-initialized scorer did not preserve every baseline span.")
    targets = {
        row["question_id"]: torch.tensor([
            temporal_iou(row["gold"], span) for span in candidates[row["question_id"]]["spans"]
        ], dtype=torch.float32)
        for row in eligible_training
    }
    args.output.mkdir(parents=True, exist_ok=True)
    epochs = 2 if args.smoke else 3 if args.train_encoder else EPOCHS
    head_lr = 0.001 if args.train_encoder else 0.01
    decay = 0.01 if args.train_encoder else 0.1
    write_json(args.output / "provenance.json", {
        "model": MODEL, "revision": REVISION, "frozen_encoder": not args.train_encoder,
        "seed": SEED, "fold": args.fold, "smoke_only": args.smoke,
        "excluded_demonstration_tids": sorted(excluded),
        "training_tids": sorted({row["transcript_id"] for row in training}),
        "training_question_ids": [row["question_id"] for row in eligible_training],
        "training_no_anchor_count": sum(not row["answer"] for row in training),
        "validation_tids": folds[args.fold],
        "context_words": 24, "prior_scale_words": 8.0,
        "anchor_marked": args.mark_anchor,
        "epochs": epochs, "learning_rate": head_lr, "weight_decay": decay,
        "encoder_learning_rate": 3e-5 if args.train_encoder else None,
        "baseline_sha256": hashlib.sha256((args.baseline / "base_legacy_questions.json").read_bytes()).hexdigest(),
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
    })
    parameters = list(head.parameters())
    groups = [{"params": parameters, "lr": head_lr}]
    if args.train_encoder:
        encoder.requires_grad_(True).train()
        encoder_parameters = list(encoder.parameters())
        groups.append({"params": encoder_parameters, "lr": 3e-5})
        parameters = [*parameters, *encoder_parameters]
        probe_name, probe_parameter = list(encoder.named_parameters())[-1]
        initial_encoder_probe = probe_parameter.detach().clone()
    optimizer = torch.optim.AdamW(groups, weight_decay=decay)
    rng = random.Random(SEED)
    losses, norms, encoder_norms = [], [], []
    started = time.monotonic()
    for epoch in range(epochs):
        order = list(eligible_training)
        rng.shuffle(order)
        optimizer.zero_grad()
        for index, row in enumerate(order):
            qid = row["question_id"]
            probability = logits(qid).softmax(dim=0)
            loss = 1.0 - torch.dot(probability, targets[qid])
            if not torch.isfinite(loss):
                raise RuntimeError("Local span risk is non-finite.")
            divisor = min(4, len(order) - (index // 4) * 4)
            (loss / divisor).backward()
            losses.append(float(loss.detach()))
            if (index + 1) % 4 == 0 or index + 1 == len(order):
                if args.train_encoder:
                    encoder_norms.append(
                        float(probe_parameter.grad.norm()) if probe_parameter.grad is not None else 0.0
                    )
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
                if not math.isfinite(norm):
                    raise RuntimeError("Local span gradients are non-finite.")
                norms.append(norm)
                optimizer.step()
                optimizer.zero_grad()
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: risk={sum(losses[-len(order):])/len(order):.6f}", flush=True)
    movement = float(head.weight.detach().norm())
    training_report = {
        "training_s": time.monotonic() - started, "head_parameters": head.weight.numel(),
        "head_update_l2": movement, "finite_nonzero_gradient": any(value > 0 for value in norms),
        "baseline_preserved_before_training": True, "encoded_cases": len(features),
        "max_encoding_s": max(encoding_times.values()), "cpu_threads": threads,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9,
        "first_epoch_risk": sum(losses[:len(eligible_training)]) / len(eligible_training),
        "last_epoch_risk": sum(losses[-len(eligible_training):]) / len(eligible_training),
        "variable_reward_cases": sum(float(values.max() - values.min()) > 0 for values in targets.values()),
        "encoder_trained": args.train_encoder,
    }
    if args.train_encoder:
        encoder_movement = float((probe_parameter.detach() - initial_encoder_probe).norm())
        training_report.update({
            "encoder_probe_parameter": probe_name, "encoder_probe_update_l2": encoder_movement,
            "encoder_probe_nonzero_gradient": any(value > 0 for value in encoder_norms),
        })
        if (
            not math.isfinite(encoder_movement) or encoder_movement <= 0
            or not all(math.isfinite(value) for value in encoder_norms)
            or not training_report["encoder_probe_nonzero_gradient"]
        ):
            raise RuntimeError("The requested encoder fine-tuning produced no finite parameter update.")
    write_json(args.output / "training.json", training_report)
    if not math.isfinite(movement) or movement <= 0 or not training_report["finite_nonzero_gradient"]:
        raise RuntimeError("Local span scorer failed its actual-update gate.")
    torch.save(head.state_dict(), args.output / "head.pt")
    if args.smoke:
        print(json.dumps({"smoke_only": True, **training_report}), flush=True)
        return 0
    if args.train_encoder:
        encoder.eval()
        encoder.save_pretrained(encoder_output, safe_serialization=True)
        training_report["encoder_checkpoint"] = str(encoder_output.resolve())
        write_json(args.output / "training.json", training_report)
    predicted, oracle = [], []
    elapsed_by_tid = {}
    with torch.no_grad():
        for row in validation:
            record = baseline_prediction(row)
            if row["answer"]:
                qid = row["question_id"]
                start = time.monotonic()
                scores = logits(qid)
                if not torch.isfinite(scores).all():
                    raise RuntimeError("Local span inference produced non-finite scores.")
                index = int(scores.argmax())
                case = candidates[qid]
                record.update({
                    "span": case["spans"][index], "local_span_candidate": index,
                    "local_span_word_range": [case["first_word"] + value for value in case["pairs"][index]],
                    "local_span_reason": "kept" if index == 0 else "edited",
                })
                preprocessing = prepared[qid]["prepare_s"] if args.train_encoder else encoding_times[qid]
                elapsed_by_tid[row["transcript_id"]] = elapsed_by_tid.get(row["transcript_id"], 0.0) + (
                    preprocessing + time.monotonic() - start
                )
                if row["label"] == 1 and row["gold"] is not None:
                    best = max(case["spans"], key=lambda span: temporal_iou(row["gold"], span))
                    oracle.append({**row, "span": best})
                else:
                    oracle.append(record)
            else:
                record["local_span_reason"] = "base_no"
                oracle.append(record)
            predicted.append(record)
    validate_coverage(predicted, validation)
    baseline = [baseline_prediction(row) for row in validation]
    request_times = {entry["transcript_id"]: entry["latency_s"] for entry in requests}
    report = {
        "pilot_only": True, "fold": args.fold, "questions": len(validation),
        "baseline": score_records(baseline), "candidate": score_records(predicted),
        "paired": paired_comparison(baseline, predicted),
        "gold_assisted_lattice_oracle": score_records(oracle),
        "oracle_is_not_achieved_performance": True,
        "changed_questions": sum(row.get("local_span_reason") == "edited" for row in predicted),
        "max_estimated_combined_s": max(
            request_times[tid] + elapsed_by_tid.get(tid, 0.0) for tid in folds[args.fold]
        ),
        "latency_note": "Separate-run CPU encoder/head sum, not a co-resident HTTP gate.",
        "training": training_report,
    }
    write_json(args.output / "questions.json", predicted)
    write_json(args.output / "summary.json", report)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
