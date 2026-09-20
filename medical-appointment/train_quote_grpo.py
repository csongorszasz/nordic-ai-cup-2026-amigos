"""Fold-isolated, geometry-rewarded quote localization from a frozen SFT adapter."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.llm_client import HFClient
from answerers.lora_data import quote_messages, split_rows
from answerers.modernbert_data import load_rows
from answerers.temporal_reward import ground_quote, quote_reward
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from train_quote_lora import evaluate
from train_quote_oof import validate_coverage
from utils import gold_evidence


MODEL = "google/gemma-4-e4b-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"


def validate_warm_start(training, validation, excluded, split, metadata, adapter_config, fold, seed):
    expected = {
        "training_question_ids": [row["question_id"] for row in training],
        "training_tids": sorted({row["transcript_id"] for row in training}),
        "validation_tids": sorted({row["transcript_id"] for row in validation}),
        "excluded_demonstration_tids": sorted(excluded),
    }
    for name, values in expected.items():
        if sorted(split[name]) != sorted(values):
            raise ValueError(f"SFT warm-start {name} does not match the exact GRPO split.")
    if (
        split["fold"] != fold or split["seed"] != seed
        or set(expected["training_tids"]) & (set(expected["validation_tids"]) | excluded)
        or any(row["label"] != 1 for row in training)
        or metadata["model"] != MODEL or metadata["revision"] != REVISION
        or metadata["smoke_only"] or not metadata["finite_training"]
        or adapter_config["base_model_name_or_path"] != MODEL
        or adapter_config["revision"] != REVISION
    ):
        raise ValueError("SFT warm-start model/training provenance is inconsistent.")


def load_inputs(baseline, sft_directory, fold, seed):
    rows = json.loads((baseline / "base_legacy_questions.json").read_text())
    requests = json.loads((baseline / "base_legacy_conversations.json").read_text())
    truth = {row["question_id"]: row for row in load_rows()}
    if (
        len(rows) != 390 or len({row["question_id"] for row in rows}) != 390
        or {row["question_id"] for row in rows} != truth.keys()
    ):
        raise ValueError("The complete, unique frozen training-corpus baseline is required.")
    for row in rows:
        reference = truth[row["question_id"]]
        gold = gold_evidence(reference)
        if (
            row["transcript_id"] != reference["transcript_id"]
            or row["question"] != reference["question"]
            or row["question_type"] != reference["question_type"]
            or row["label"] != int(reference["label"])
            or row["gold"] != (list(gold) if gold is not None else None)
            or not isinstance(row["answer"], bool) or row["prediction"] != int(row["answer"])
        ):
            raise ValueError(f"Baseline provenance mismatch for {row['question_id']}.")
    tids = {row["transcript_id"] for row in rows}
    if len(requests) != len(tids) or {entry["transcript_id"] for entry in requests} != tids:
        raise ValueError("Baseline request manifest is incomplete or duplicated.")
    excluded = {tid for entry in requests for tid in entry["demonstration_tids"]}
    training, validation, _ = split_rows(rows, excluded, fold, seed)
    split = json.loads((sft_directory / "split.json").read_text())
    metadata = json.loads((sft_directory / "training.json").read_text())
    config = json.loads((sft_directory / "adapter" / "adapter_config.json").read_text())
    validate_warm_start(training, validation, excluded, split, metadata, config, fold, seed)
    transcripts = {}
    for entry in requests:
        tid = entry["transcript_id"]
        transcript = json.loads((baseline / "transcripts" / f"{tid}.json").read_text())
        digest = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
        if digest != entry["transcript_sha256"]:
            raise ValueError(f"Frozen transcript content changed for {tid}.")
        transcripts[tid] = transcript
    return training, validation, transcripts, requests, split


def prompt_record(row, transcript, tokenizer):
    rendered = tokenizer.apply_chat_template(
        quote_messages(row["question"], transcript), tokenize=False, add_generation_prompt=True,
    )
    expected = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if len(expected) + 128 > 4096:
        raise ValueError(f"Do not truncate the evidence context: {row['question_id']}.")
    return {"prompt": rendered, "question_id": row["question_id"]}, expected


def configure_completion_eos(tokenizer, native_eos):
    """TRL masks against one EOS; use the SFT assistant-turn terminator."""
    messages = [{"role": "user", "content": "Return JSON."}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": "{}"}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Assistant template is not an exact generation continuation.")
    native_eos = native_eos if isinstance(native_eos, (list, tuple)) else [native_eos]
    stops = [token for token in full_ids[len(prefix_ids):] if token in native_eos]
    if not stops:
        raise ValueError("SFT assistant turn has no native model stopping token.")
    tokenizer.eos_token = tokenizer.convert_ids_to_tokens(stops[0])
    if tokenizer.eos_token_id != stops[0]:
        raise ValueError("Tokenizer did not accept the assistant-turn EOS.")
    return stops[0]


class GroundingReward:
    def __init__(self, rows, transcripts, generations, journal):
        self.rows = {row["question_id"]: row for row in rows}
        if len(self.rows) != len(rows) or any(row["label"] != 1 for row in rows):
            raise ValueError("Reward inputs must be unique training positives.")
        self.transcripts = transcripts
        self.generations = generations
        self.journal = journal
        self.records = []
        self.groups = 0
        self.variable_groups = 0

    def __call__(self, completions, question_id, trainer_state=None, log_metric=None, **kwargs):
        if not completions or len(completions) != len(question_id) or len(completions) % self.generations:
            raise ValueError("Reward batch does not contain complete generation groups.")
        for start in range(0, len(question_id), self.generations):
            if len(set(question_id[start:start + self.generations])) != 1:
                raise ValueError("A generation group mixes training questions.")
        batch = []
        for completion, qid in zip(completions, question_id):
            if qid not in self.rows:
                raise ValueError(f"Reward requested for non-training question {qid}.")
            row = self.rows[qid]
            raw = completion
            if isinstance(raw, list):
                raw = (
                    raw[0].get("content") if len(raw) == 1
                    and isinstance(raw[0], dict) and raw[0].get("role") == "assistant" else None
                )
            result = quote_reward(raw, self.transcripts[row["transcript_id"]], row["gold"])
            batch.append({
                "question_id": qid, "step": trainer_state.global_step if trainer_state is not None else 0,
                "raw": raw, "span": result.grounding.span, "reason": result.grounding.reason,
                "tiou": result.tiou, "wasserstein": result.wasserstein,
                "validity": result.validity, "reward": result.total,
            })
        for start in range(0, len(batch), self.generations):
            values = [entry["reward"] for entry in batch[start:start + self.generations]]
            self.groups += 1
            self.variable_groups += max(values) - min(values) > 1e-8
        if self.journal is not None:
            with self.journal.open("a", encoding="utf-8") as handle:
                for entry in batch:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
        self.records.extend(batch)
        if log_metric is not None:
            for key in ("tiou", "wasserstein", "validity"):
                log_metric(f"grounding/{key}", sum(entry[key] for entry in batch) / len(batch))
        return [entry["reward"] for entry in batch]

    def summary(self):
        return {
            "rollouts": len(self.records), "groups": self.groups,
            "variable_reward_groups": self.variable_groups,
            "grounded_rollouts": sum(entry["reason"] == "grounded" for entry in self.records),
            "positive_iou_rollouts": sum(entry["tiou"] > 0 for entry in self.records),
            "mean_tiou": sum(entry["tiou"] for entry in self.records) / len(self.records)
            if self.records else None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sft-directory", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/grpo_pilot"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    training, validation, transcripts, requests, split = load_inputs(
        args.baseline, args.sft_directory, args.fold, args.seed,
    )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("GRPO output already contains a run; use a fresh directory.")
    adapter = args.sft_directory / "adapter"
    adapter_hash = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    saved_sft = json.loads((args.sft_directory / "adapted_questions.json").read_text())
    validate_coverage(saved_sft, validation)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "split.json", split)

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GRPO requires exactly its one allocated GPU.")
    set_seed(args.seed)
    client = HFClient(model_name=MODEL, revision=REVISION, dtype="bfloat16",
                      max_new_tokens=128, legacy_special_tokens=False)
    client.warm_up()
    model = PeftModel.from_pretrained(
        client._model, str(adapter), is_trainable=True, local_files_only=True,
    )
    client._model = model
    inference_generation = deepcopy(model.generation_config)
    inference_eos = client._tokenizer.eos_token
    records, expected_ids = [], {}
    for row in training:
        record, ids = prompt_record(row, transcripts[row["transcript_id"]], client._tokenizer)
        records.append(record)
        expected_ids[row["question_id"]] = ids
    if args.smoke:
        records = sorted(records, key=lambda row: len(expected_ids[row["question_id"]]), reverse=True)[:8]
    reward_ids = {record["question_id"] for record in records}
    reward = GroundingReward(
        [row for row in training if row["question_id"] in reward_ids],
        transcripts, 4, args.output / "rollouts.jsonl",
    )
    request_times = {entry["transcript_id"]: entry["latency_s"] for entry in requests}
    if not args.smoke:
        before, _ = evaluate(client, validation, transcripts, request_times, args.output, "sft_replay")
        validate_coverage(before, validation)
    completion_eos = configure_completion_eos(client._tokenizer, model.generation_config.eos_token_id)
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not initial or any("lora_" not in name or ".default." not in name for name in initial):
        raise RuntimeError("Only the pretrained default LoRA adapter may be optimized.")

    class GradientGate(TrainerCallback):
        def __init__(self):
            self.norms = []

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            squares = [
                parameter.grad.detach().float().square().sum().item()
                for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None
            ]
            norm = math.sqrt(sum(squares))
            if not math.isfinite(norm):
                raise RuntimeError("GRPO produced non-finite adapter gradients.")
            self.norms.append(norm)

    gradients = GradientGate()
    config = GRPOConfig(
        output_dir=str(args.output / "trainer"),
        seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4, max_completion_length=128,
        num_train_epochs=1, max_steps=8 if args.smoke else -1,
        learning_rate=1e-5, lr_scheduler_type="constant", warmup_steps=0,
        optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        temperature=1.0, top_p=1.0, top_k=0,
        beta=0.02, loss_type="dr_grpo", scale_rewards="none",
        num_iterations=1, mask_truncated_completions=True,
        generation_kwargs={"use_cache": True, "num_beams": 1},
        use_vllm=False, disable_dropout=True,
        remove_unused_columns=False, shuffle_dataset=not args.smoke,
        save_strategy="no", eval_strategy="no",
        logging_strategy="steps", logging_steps=1 if args.smoke else 10,
        report_to="none", push_to_hub=False, disable_tqdm=True,
    )
    trainer = GRPOTrainer(
        model=model, args=config, processing_class=client._tokenizer,
        reward_funcs=reward, train_dataset=Dataset.from_list(records), callbacks=[gradients],
    )
    actual_ids, _, _ = trainer._tokenize_prompts([record["prompt"] for record in records])
    if actual_ids != [expected_ids[record["question_id"]] for record in records]:
        raise ValueError("GRPO tokenization differs from the SFT/serving prompt; do not train.")
    reference = {
        name: parameter for name, parameter in model.named_parameters() if ".ref." in name
    }
    if len(reference) != len(initial) or any(parameter.requires_grad for parameter in reference.values()):
        raise RuntimeError("GRPO must use a frozen copy of the SFT adapter as its KL reference.")
    for name, value in initial.items():
        ref = reference[name.replace(".default.", ".ref.")]
        if not torch.equal(ref.detach().cpu(), value):
            raise RuntimeError("The KL reference is not the SFT warm start.")
    write_json(args.output / "provenance.json", {
        "model": MODEL, "revision": REVISION,
        "sft_directory": str(args.sft_directory.resolve()), "adapter_sha256": adapter_hash,
        "baseline_sha256": hashlib.sha256((args.baseline / "base_legacy_questions.json").read_bytes()).hexdigest(),
        "smoke_only": args.smoke, "training_question_ids": sorted(reward_ids),
        "prompt_tokens_match": True, "frozen_sft_reference": True,
        "completion_eos_token_id": completion_eos,
        "runtime": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl")},
        "training_arguments": config.to_dict(),
    })
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1e9
    parameters = dict(model.named_parameters())
    delta = sum(
        (parameters[name].detach().cpu().float() - value.float()).square().sum().item()
        for name, value in initial.items()
    )
    for name, value in initial.items():
        if not torch.equal(reference[name.replace(".default.", ".ref.")].detach().cpu(), value):
            raise RuntimeError("Training modified the frozen SFT reference.")
    report = {
        **reward.summary(), "smoke_only": args.smoke, "fold": args.fold, "seed": args.seed,
        "training_s": time.monotonic() - started, "peak_cuda_gb": peak,
        "trainable_parameters": sum(value.numel() for value in initial.values()),
        "adapter_update_l2": math.sqrt(delta), "gradient_norms": gradients.norms,
        "train_metrics": result.metrics,
    }
    write_json(args.output / "training.json", report)
    if (
        not math.isfinite(result.training_loss) or not math.isfinite(delta) or delta <= 0
        or not any(value > 0 for value in gradients.norms) or reward.variable_groups == 0
        or not report["grounded_rollouts"] or not report["positive_iou_rollouts"]
    ):
        raise RuntimeError("GRPO failed the finite-update, reward-signal, or grounding gate.")
    model.gradient_checkpointing_disable()
    model.set_adapter("default")
    model.eval()
    client._tokenizer.eos_token = inference_eos
    model.generation_config = inference_generation
    model.save_pretrained(args.output / "adapter", selected_adapters=["default"], safe_serialization=True)
    if args.smoke:
        row = reward.rows[records[0]["question_id"]]
        raw = client.generate(
            quote_messages(row["question"], transcripts[row["transcript_id"]]),
            max_new_tokens=128, deadline=time.monotonic() + 20,
        )
        grounded = ground_quote(raw, transcripts[row["transcript_id"]])
        report.update({"generation_probe_question": row["question_id"],
                       "generation_probe_raw": raw, "generation_probe_reason": grounded.reason})
        write_json(args.output / "training.json", report)
        if grounded.span is None:
            raise RuntimeError("Post-update greedy generation failed the training-probe grounding gate.")
    else:
        after, timings = evaluate(client, validation, transcripts, request_times, args.output, "grpo")
        validate_coverage(after, validation)
        baseline = [
            {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row["duration"]) if row["answer"] else None}
            for row in validation
        ]
        report = {
            "pilot_only": True, "fold": args.fold, "questions": len(validation),
            "baseline": score_records(baseline), "sft": score_records(before), "grpo": score_records(after),
            "versus_baseline": paired_comparison(baseline, after),
            "versus_sft": paired_comparison(before, after),
            "sft_runtime_drift": paired_comparison(saved_sft, before),
            "reasons": dict(Counter(row["localizer_reason"] for row in after)),
            "max_estimated_combined_s": max(entry["estimated_combined_s"] for entry in timings),
            "latency_note": "Separate-run sum, not an HTTP gate.",
            "training": report,
        }
        write_json(args.output / "summary.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
