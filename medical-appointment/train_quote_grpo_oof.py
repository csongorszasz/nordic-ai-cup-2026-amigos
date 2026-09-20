"""Complete fixed GRPO folds serially, reusing the unchanged held-out pilot."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.lora_data import split_rows
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from train_quote_grpo import load_inputs
from train_quote_oof import FROZEN_SOURCES as SFT_SOURCES, validate_coverage


ROOT = Path(__file__).resolve().parent
FROZEN_SOURCES = (
    *SFT_SOURCES, "train_quote_oof.py", "train_quote_grpo.py",
    "answerers/temporal_reward.py", "requirements-grpo.txt", "requirements-lora.txt",
)
FIXED_ARGUMENTS = {
    "seed": 13, "data_seed": 13, "num_train_epochs": 1, "max_steps": -1,
    "learning_rate": 1e-5, "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4, "num_generations": 4,
    "max_completion_length": 128, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
    "beta": 0.02, "loss_type": "dr_grpo", "scale_rewards": "none",
    "num_iterations": 1, "mask_truncated_completions": True,
    "bf16": True, "fp16": False, "weight_decay": 0.0,
    "use_vllm": False, "disable_dropout": True,
}


def verify_sources(pilot_run, root=ROOT):
    manifest = json.loads((pilot_run / "source_manifest.json").read_text())
    for name in FROZEN_SOURCES:
        digest = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if manifest["files"].get(name) != digest:
            raise ValueError(f"Pilot training/inference source changed: {name}")


def verify_provenance(provenance, baseline_digest, runtime):
    if (
        provenance["smoke_only"] or not provenance["prompt_tokens_match"]
        or not provenance["frozen_sft_reference"]
        or provenance["baseline_sha256"] != baseline_digest
        or provenance["runtime"] != runtime
    ):
        raise ValueError("GRPO pilot/fold runtime or input provenance changed.")
    for name, expected in FIXED_ARGUMENTS.items():
        if provenance["training_arguments"].get(name) != expected:
            raise ValueError(f"Fixed GRPO hyperparameter changed: {name}")


def pilot_arguments(request, baseline):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sft-directory", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/grpo_pilot"))
    args = parser.parse_args(request["arguments"])
    if (
        request["script"] != "train_quote_grpo.py" or args.smoke
        or args.fold != 0 or args.seed != 13 or args.baseline.resolve() != baseline.resolve()
        or args.output != Path("results/grpo_pilot")
    ):
        raise ValueError("Pilot request does not match the fixed complete OOF experiment.")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--sft-oof-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/grpo_oof"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("OOF output already contains a run; use a fresh directory.")
    verify_sources(args.pilot_run)
    pilot_request = json.loads((args.pilot_run / "run_request.json").read_text())
    pilot_args = pilot_arguments(pilot_request, args.baseline)
    rows = json.loads((args.baseline / "base_legacy_questions.json").read_text())
    requests = json.loads((args.baseline / "base_legacy_conversations.json").read_text())
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    expected = [row for row in rows if row["transcript_id"] not in excluded]
    _, _, folds = split_rows(rows, excluded, 0, 13)
    baseline_digest = hashlib.sha256((args.baseline / "base_legacy_questions.json").read_bytes()).hexdigest()
    runtime = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "trl")}
    pilot_directory = args.pilot_run / "results" / "grpo_pilot"
    provenance = json.loads((pilot_directory / "provenance.json").read_text())
    verify_provenance(provenance, baseline_digest, runtime)
    sft_directories = [
        pilot_args.sft_directory,
        *(args.sft_oof_run / "results" / "lora_oof" / f"fold_{index}" for index in range(1, len(folds))),
    ]
    adapter_hashes = []
    for index, directory in enumerate(sft_directories):
        load_inputs(args.baseline, directory, index, 13)
        adapter_hashes.append(hashlib.sha256(
            (directory / "adapter" / "adapter_model.safetensors").read_bytes()
        ).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    adapted, sft, summaries = [], [], []
    for index, held_out in enumerate(folds):
        directory = pilot_directory if index == 0 else args.output / f"fold_{index}"
        if index:
            with (args.output / f"fold_{index}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable, str(ROOT / "train_quote_grpo.py"),
                        "--baseline", str(args.baseline),
                        "--sft-directory", str(sft_directories[index]),
                        "--fold", str(index), "--seed", "13", "--output", str(directory),
                    ],
                    check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
        split = json.loads((directory / "split.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        verify_provenance(provenance, baseline_digest, runtime)
        if (
            split["fold"] != index or split["seed"] != 13
            or set(split["validation_tids"]) != set(held_out)
            or set(split["training_tids"]) & (set(held_out) | excluded)
            or split["excluded_demonstration_tids"] != sorted(excluded)
            or provenance["adapter_sha256"] != adapter_hashes[index]
        ):
            raise ValueError(f"Fold {index} training provenance is inconsistent.")
        held_rows = [row for row in expected if row["transcript_id"] in held_out]
        after = json.loads((directory / "grpo_questions.json").read_text())
        before = json.loads((directory / "sft_replay_questions.json").read_text())
        validate_coverage(after, held_rows)
        validate_coverage(before, held_rows)
        adapted.extend(after)
        sft.extend(before)
        report = json.loads((directory / "summary.json").read_text())
        summaries.append({
            "fold": index, "baseline": report["baseline"], "sft": report["sft"], "grpo": report["grpo"],
            "versus_sft": report["versus_sft"], "sft_runtime_drift": report["sft_runtime_drift"],
            "reasons": report["reasons"], "max_estimated_combined_s": report["max_estimated_combined_s"],
        })
        write_json(args.output / "grpo_oof.json", adapted)
        write_json(args.output / "sft_oof.json", sft)
        print(json.dumps(summaries[-1]), flush=True)
    validate_coverage(adapted, expected)
    validate_coverage(sft, expected)
    baseline = [
        {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row["duration"]) if row["answer"] else None}
        for row in expected
    ]
    summary = {
        "complete_oof": True, "questions": len(expected), "folds": summaries,
        "excluded_demonstration_tids": sorted(excluded),
        "baseline": score_records(baseline), "sft": score_records(sft), "grpo": score_records(adapted),
        "versus_baseline": paired_comparison(baseline, adapted),
        "versus_sft": paired_comparison(sft, adapted),
        "runtime": runtime, "max_estimated_combined_s": max(
            entry["max_estimated_combined_s"] for entry in summaries
        ),
        "latency_note": "Separate-run sum, not an HTTP acceptance measurement.",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "folds"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
