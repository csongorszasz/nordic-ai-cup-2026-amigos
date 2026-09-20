"""Feasibility-gated E4B quote-localizer pilot; never changes base decisions."""

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

from answerers.align import align_quote_matches
from answerers.boundaries import adjusted_span
from answerers.base import normalize_answer
from answerers.llm_client import HFClient
from answerers.llm_parse import extract_json
from answerers.lora_data import quote_messages, split_rows, training_example
from answerers.modernbert_data import load_rows
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from utils import gold_evidence

logger = logging.getLogger(__name__)


def evaluate(client, rows, transcripts, request_times, output, tag):
    by_tid = {}
    for row in rows:
        by_tid.setdefault(row["transcript_id"], []).append(row)
    records, timings = [], []
    for tid, group in by_tid.items():
        started = time.monotonic()
        deadline = started + 20.0
        transcript = transcripts[tid]
        for row in group:
            span = adjusted_span(row["span"], (0.2, 0.0), transcript["duration"]) if row["answer"] else None
            reason, quote, raw = "base_no", None, None
            if row["answer"]:
                reason = "budget_keep"
                if time.monotonic() < deadline:
                    try:
                        raw = client.generate(
                            quote_messages(row["question"], transcript),
                            max_new_tokens=128, deadline=deadline,
                        )
                        payload = extract_json(raw)
                        quote = payload.get("evidence_quote") if isinstance(payload, dict) else None
                        matches = align_quote_matches(
                            transcript["words"], quote if isinstance(quote, str) else ""
                        )
                        if len(matches) == 1:
                            proposal = adjusted_span(matches[0][:2], (0.2, 0.0), transcript["duration"])
                            valid, checked = normalize_answer(
                                (True, proposal), duration=transcript["duration"],
                                context=row["question_id"],
                            )
                            if valid:
                                span, reason = list(checked), "localized"
                            else:
                                reason = "bounds_keep"
                        else:
                            reason = "alignment_keep"
                    except Exception:
                        logger.exception("Localizer failed for %s; preserving baseline.", row["question_id"])
                        reason = "error_keep"
            records.append({
                **row, "span": span, "localizer_reason": reason,
                "localizer_quote": quote, "localizer_raw": raw,
            })
        elapsed = time.monotonic() - started
        timings.append({"transcript_id": tid, "localizer_s": elapsed,
                        "estimated_combined_s": request_times[tid] + elapsed})
        write_json(output / f"{tag}_questions.json", records)
        print(f"{tag} {tid}: {elapsed:.2f}s", flush=True)
    return records, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-4-e4b-it")
    parser.add_argument("--revision", default="ee0ef6023621cff504d758262d4e04895a5af4a2")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/lora_pilot"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    if not torch.cuda.is_available():
        raise RuntimeError("This training pilot requires its allocated GPU.")
    if args.epochs < 1 or args.accumulation < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Training limits must be positive and finite.")
    torch.manual_seed(args.seed)
    rows = json.loads((args.baseline / "base_legacy_questions.json").read_text())
    requests = json.loads((args.baseline / "base_legacy_conversations.json").read_text())
    if len(rows) != 390 or len({row["question_id"] for row in rows}) != 390:
        raise ValueError("The complete frozen baseline is required.")
    truth = {row["question_id"]: row for row in load_rows()}
    for row in rows:
        reference = truth[row["question_id"]]
        gold = gold_evidence(reference)
        if (
            row["question"] != reference["question"]
            or row["label"] != int(reference["label"])
            or row["gold"] != (list(gold) if gold is not None else None)
        ):
            raise ValueError(f"Training provenance mismatch for {row['question_id']}.")
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    training, validation, folds = split_rows(rows, excluded, args.fold, args.seed)
    transcripts = {
        request["transcript_id"]: json.loads(
            (args.baseline / "transcripts" / f"{request['transcript_id']}.json").read_text()
        ) for request in requests
    }
    request_times = {request["transcript_id"]: request["latency_s"] for request in requests}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "split.json", {
        "excluded_demonstration_tids": sorted(excluded),
        "training_tids": sorted({row["transcript_id"] for row in training}),
        "validation_tids": folds[args.fold],
        "training_question_ids": [row["question_id"] for row in training],
        "fold": args.fold, "seed": args.seed,
    })
    client = HFClient(model_name=args.model, revision=args.revision, dtype="bfloat16",
                      max_new_tokens=128, legacy_special_tokens=False)
    client.warm_up()
    features = [
        training_example(row, transcripts[row["transcript_id"]], client._tokenizer)
        for row in training
    ]
    baseline = [
        {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row.get("duration")) if row["answer"] else None}
        for row in validation
    ]
    if not args.smoke:
        unadapted, before_times = evaluate(client, validation, transcripts, request_times, args.output, "unadapted")
    targets = [
        name for name, module in client._model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".self_attn." in name
        and name.endswith((".q_proj", ".v_proj"))
        and "vision" not in name and "audio" not in name
    ]
    if not targets:
        raise RuntimeError("No language attention projections matched the adapter plan.")
    model = get_peft_model(
        client._model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
            lora_dropout=0.05, target_modules=targets, bias="none", revision=args.revision,
        ),
    )
    client._model = model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise RuntimeError("A base-model parameter was unexpectedly made trainable.")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    began = time.monotonic()
    rng = random.Random(args.seed)
    for epoch in range(1 if args.smoke else args.epochs):
        order = list(range(len(features)))
        rng.shuffle(order)
        if args.smoke:
            order = [max(order, key=lambda index: len(features[index]["input_ids"]))]
        for step, index in enumerate(order):
            tensors = {
                name: torch.tensor([value], dtype=torch.long, device=client._device)
                for name, value in features[index].items()
            }
            outputs = model(**tensors, use_cache=False)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            group_start = step // args.accumulation * args.accumulation
            divisor = 1 if args.smoke else min(args.accumulation, len(order) - group_start)
            (loss / divisor).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % args.accumulation == 0 or step + 1 == len(order) or args.smoke:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Adapter gradients are absent or non-finite.")
                optimizer.step()
                optimizer.zero_grad()
            del outputs, loss, tensors
        print(f"epoch {epoch + 1}: mean_loss={sum(losses[-len(order):])/len(order):.5f}", flush=True)
    smoke = {
        "finite_training": True, "examples": len(features),
        "max_sequence_tokens": max(len(item["input_ids"]) for item in features),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "target_modules": targets, "peak_cuda_gb": torch.cuda.max_memory_allocated() / 1e9,
        "training_s": time.monotonic() - began, "loss_first": losses[0], "loss_last": losses[-1],
        "smoke_only": args.smoke, "model": args.model, "revision": args.revision,
    }
    write_json(args.output / "training.json", smoke)
    if args.smoke:
        model.gradient_checkpointing_disable()
        model.eval()
        probe = training[max(range(len(features)), key=lambda index: len(features[index]["input_ids"]))]
        raw = client.generate(
            quote_messages(probe["question"], transcripts[probe["transcript_id"]]),
            max_new_tokens=128, deadline=time.monotonic() + 20,
        )
        parsed = extract_json(raw)
        quote = parsed.get("evidence_quote") if isinstance(parsed, dict) else None
        if not isinstance(quote, str) or not align_quote_matches(
            transcripts[probe["transcript_id"]]["words"], quote
        ):
            raise RuntimeError("Adapted generation did not produce a grounded training-probe quote.")
        smoke["adapted_generation_valid"] = True
        smoke["generation_probe_question"] = probe["question_id"]
        smoke["generation_probe_raw"] = raw
        write_json(args.output / "training.json", smoke)
        print(json.dumps({key: value for key, value in smoke.items() if key != "target_modules"}), flush=True)
        return 0
    model.gradient_checkpointing_disable()
    model.eval()
    model.save_pretrained(args.output / "adapter", safe_serialization=True)
    adapted, after_times = evaluate(client, validation, transcripts, request_times, args.output, "adapted")
    reasons = {}
    for row in adapted:
        reasons[row["localizer_reason"]] = reasons.get(row["localizer_reason"], 0) + 1
    report = {
        "pilot_only": True, "fold": args.fold, "seed": args.seed,
        "baseline": score_records(baseline), "unadapted": score_records(unadapted),
        "adapted": score_records(adapted),
        "versus_baseline": paired_comparison(baseline, adapted),
        "versus_unadapted": paired_comparison(unadapted, adapted),
        "training": smoke,
        "adapted_reasons": reasons,
        "max_estimated_combined_s": max(item["estimated_combined_s"] for item in after_times),
        "latency_note": "Separate-run sum, not an HTTP gate.",
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "training"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
