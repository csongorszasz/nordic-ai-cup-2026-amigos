"""Use each frozen outer-fold adapter's mean completion NLL to rank two quotes."""

import argparse
import json
import math
import time
from pathlib import Path

from answerers.align import align_quote_matches
from answerers.boundaries import adjusted_span
from answerers.llm_client import HFClient
from answerers.lora_data import encode_completion, quote_messages
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from train_quote_oof import validate_coverage


def canonical_quote(words, quote, span, duration):
    if not isinstance(quote, str) or span is None:
        return None
    for match in align_quote_matches(words, quote):
        corrected = adjusted_span(match[:2], (0.2, 0.0), duration)
        if max(abs(a - b) for a, b in zip(corrected, span)) < 1e-6:
            return " ".join("".join(word["word"] for word in words[match[2]:match[3] + 1]).split())
    return None


def prefer_adapter(base_loss, adapter_loss):
    if not math.isfinite(base_loss) or not math.isfinite(adapter_loss):
        raise ValueError("Quote likelihood must be finite.")
    return adapter_loss < base_loss - 1e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/adapter_likelihood"))
    args = parser.parse_args()
    import torch
    from peft import PeftModel

    summary = json.loads((args.oof / "summary.json").read_text())
    if not summary.get("complete_oof"):
        raise ValueError("Complete OOF predictions are required.")
    original = json.loads((args.baseline / "base_legacy_questions.json").read_text())
    excluded = set(summary["excluded_demonstration_tids"])
    selected = [row for row in original if row["transcript_id"] not in excluded]
    baseline = [
        {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row.get("duration")) if row["answer"] else None}
        for row in selected
    ]
    alternatives = json.loads((args.oof / "adapted_oof.json").read_text())
    validate_coverage(alternatives, selected)
    by_id = {row["question_id"]: row for row in alternatives}
    transcripts = {
        tid: json.loads((args.baseline / "transcripts" / f"{tid}.json").read_text())
        for tid in {row["transcript_id"] for row in baseline}
    }
    client = HFClient(
        model_name="google/gemma-4-e4b-it", revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        dtype="bfloat16", legacy_special_tokens=False,
    )
    client._load()
    model = None
    predictions, logs = [], []
    for fold in range(5):
        folder = args.pilot if fold == 0 else args.oof / f"fold_{fold}"
        split = json.loads((folder / "split.json").read_text())
        held = set(split["validation_tids"])
        if held & (set(split["training_tids"]) | excluded):
            raise ValueError("Adapter provenance overlaps held-out or demonstration data.")
        adapter_name = f"fold{fold}"
        if model is None:
            model = PeftModel.from_pretrained(
                client._model, str(folder / "adapter"), adapter_name=adapter_name,
                is_trainable=False,
            )
        else:
            model.load_adapter(str(folder / "adapter"), adapter_name=adapter_name, is_trainable=False)
        model.set_adapter(adapter_name)
        model.eval()
        if model.active_adapters != [adapter_name]:
            raise ValueError("Unexpected active adapter mixture.")

        def nll(messages, quote):
            feature = encode_completion(
                client._tokenizer, messages, json.dumps({"evidence_quote": quote})
            )
            tensors = {
                name: torch.tensor([values], dtype=torch.long, device=client._device)
                for name, values in feature.items()
            }
            with torch.no_grad():
                output = model(**tensors, use_cache=False)
            loss = float(output.loss)
            del output, tensors
            return loss

        changed = scored = 0
        for row in baseline:
            if row["transcript_id"] not in held:
                continue
            other = by_id[row["question_id"]]
            result = {**row, "reranked": False}
            if row["answer"] and other["span"] != row["span"]:
                transcript = transcripts[row["transcript_id"]]
                first = canonical_quote(
                    transcript["words"], row.get("quote"), row["span"], transcript["duration"]
                )
                second = canonical_quote(
                    transcript["words"], other.get("localizer_quote"), other["span"], transcript["duration"]
                )
                if first is None or second is None:
                    result["reason"] = "unmatched_quote_keep"
                else:
                    started = time.monotonic()
                    messages = quote_messages(row["question"], transcript)
                    base_loss, other_loss = nll(messages, first), nll(messages, second)
                    accepted = prefer_adapter(base_loss, other_loss)
                    scored += 1
                    if accepted:
                        result["span"] = other["span"]
                        result["reranked"] = True
                        changed += 1
                    logs.append({
                        "question_id": row["question_id"], "transcript_id": row["transcript_id"],
                        "fold": fold, "base_mean_nll": base_loss, "adapter_mean_nll": other_loss,
                        "accepted": accepted, "scoring_s": time.monotonic() - started,
                    })
            predictions.append(result)
        write_json(args.output / "questions.json", predictions)
        write_json(args.output / "likelihoods.json", logs)
        print(f"fold {fold}: compared={scored}, changed={changed}", flush=True)
    validate_coverage(predictions, selected)
    report = {
        "complete_oof": True, "fit_policy_on_oof": False,
        "baseline": score_records(baseline), "candidate": score_records(predictions),
        "paired": paired_comparison(baseline, predictions),
        "compared_questions": len(logs),
        "changed_questions": sum(row["reranked"] for row in predictions),
        "max_scoring_s": max((row["scoring_s"] for row in logs), default=0.0),
        "total_scoring_s": sum(row["scoring_s"] for row in logs),
        "latency_note": "Scoring overhead only; not an HTTP acceptance gate.",
    }
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
