"""Paired prompt-only probes on frozen inputs, with a locked confirmation stage."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers.llm import LLMAnswerer
from answerers.llm_client import DEVICE, HFClient
from answerers.llm_prompt import (
    build_l1_messages, reformat_frozen_demonstrations, schema_hint, serialize_transcript, system_prompt,
)
from audit_localization import exact_score
from benchmark import RecordingClient, paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs
from prepare_prompt_round import prompt_hash, round_split, stratum


ARMS = ("base", "v1", "v1_claim")
PIPELINE_FILES = (
    "answerers/llm_prompt.py", "answerers/llm_client.py", "answerers/llm.py",
    "answerers/llm_parse.py", "answerers/align.py", "answerers/base.py", "answerers/boundaries.py",
    "probe_prompt_round.py",
)


def definition_hash(variant):
    if variant not in ARMS:
        raise ValueError("This prompt round permits only its two declared candidates and control.")
    return hashlib.sha256((system_prompt(variant) + "\n" + schema_hint(variant)).encode()).hexdigest()


def validate_inputs(rows, requests, transcripts, manifest, frozen):
    split = round_split(rows, requests)
    if any(manifest.get(key) != value for key, value in split.items()):
        raise ValueError("Prompt-round conversation partitions changed after preparation.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["transcript_id"]].append(row)
    if set(frozen) != set(grouped):
        raise ValueError("The frozen prompt inputs do not cover every conversation.")
    for request in requests:
        tid = request["transcript_id"]
        entry = frozen[tid]
        if (
            entry["question_ids"] != [row["question_id"] for row in grouped[tid]]
            or entry["demonstration_tids"] != request["demonstration_tids"]
        ):
            raise ValueError(f"The frozen question order or demonstration sources changed: {tid}")
        messages = build_l1_messages(
            transcripts[tid], [row["question"] for row in grouped[tid]], entry["few_shot"], variant="base",
        )
        digest = prompt_hash(messages)
        if len(request["calls"]) != 1 or digest != request["calls"][0]["prompt_sha256"] or digest != entry["prompt_sha256"]:
            raise ValueError(f"The recorded incumbent messages no longer reproduce: {tid}")
    return grouped


def selected_tids(manifest, phase):
    if phase == "feasibility":
        tid = manifest["feasibility_tid"]
        if tid not in manifest["development_tids"]:
            raise ValueError("Feasibility must not consume a confirmation conversation.")
        return [tid]
    if phase == "pilot":
        if not set(manifest["pilot_tids"]).issubset(manifest["development_tids"]):
            raise ValueError("The pilot escaped the development partition.")
        return list(manifest["pilot_tids"])
    if phase in {"development", "confirmation"}:
        return list(manifest[f"{phase}_tids"])
    raise ValueError("Unknown prompt-round phase.")


def validate_confirmation(selection, development, manifest, signature, source_hashes):
    variant = selection.get("variant")
    if (
        variant not in ARMS[1:] or selection.get("semantic_review_complete") is not True
        or selection.get("signature") != signature
        or selection.get("definition_sha256") != definition_hash(variant)
        or development.get("phase") != "development" or development.get("complete") is not True
        or development.get("selected_tids") != manifest["development_tids"]
        or development.get("signature") != signature or development.get("pipeline_source_sha256") != source_hashes
        or development.get("definition_sha256", {}).get(variant) != definition_hash(variant)
        or variant not in development.get("variants", {}) or "base" not in development.get("variants", {})
        or development["variants"][variant].get("failed_questions") != 0
        or development["variants"]["base"].get("failed_questions") != 0
    ):
        raise ValueError("Confirmation requires a frozen, semantically reviewed candidate from this exact development run.")
    return ["base", variant]


def records_for(group, answers, duration):
    if len(answers) != len(group):
        raise ValueError("The answerer omitted or duplicated prompt-round questions.")
    records = []
    for row, (answer, span, info) in zip(group, answers):
        raw_span = list(span) if span is not None else None
        final_span = adjusted_span(span, (0.2, 0.0), duration) if answer else None
        records.append({
            **row, "answer": answer, "prediction": int(answer),
            "raw_span": raw_span, "span": final_span, "proposed_span": raw_span,
            "original_span": raw_span, "calibration_applied": final_span != raw_span,
            "raw_answer": info.get("raw_answer", answer),
            "quote": info.get("quote"), "word_range": info.get("word_range"),
            "decided_by": info.get("decided_by", "unknown"),
        })
    return records


def matched_analysis_rows(reference, control, candidate):
    maps = [{row["question_id"]: row for row in rows} for rows in (reference, control, candidate)]
    if (
        any(len(mapping) != len(rows) for mapping, rows in zip(maps, (reference, control, candidate)))
        or not maps[0].keys() == maps[1].keys() == maps[2].keys()
    ):
        raise ValueError("Prompt diagnostics require identical complete unique question sets.")
    triples = []
    for qid, row in maps[0].items():
        base, proposed = maps[1][qid], maps[2][qid]
        if any(row[key] != other[key] for other in (base, proposed) for key in (
            "transcript_id", "question", "question_type", "label", "gold",
        )):
            raise ValueError("Prompt diagnostics cannot compare different questions or references.")
        triples.append((row, base, proposed))
    return triples


def localization_strata(reference, control, candidate):
    groups = defaultdict(list)
    for original, base, proposed in matched_analysis_rows(reference, control, candidate):
        groups[stratum(original)].append((base, proposed))
    results = {}
    for name, pairs in groups.items():
        base_score = exact_score([base for base, _ in pairs])
        candidate_score = exact_score([proposed for _, proposed in pairs])
        results[name] = {
            "control": base_score, "candidate": candidate_score,
            "composite_delta": candidate_score["score"] - base_score["score"],
            "mean_tiou_delta": candidate_score["mean_tiou"] - base_score["mean_tiou"],
            "decision_changes": sum(base["answer"] != proposed["answer"] for base, proposed in pairs),
            "citation_changes": sum(base["span"] != proposed["span"] for base, proposed in pairs),
        }
    return results


def semantic_changes(control, candidate):
    cases = []
    for _, base, proposed in matched_analysis_rows(control, control, candidate):
        if all(base[key] == proposed[key] for key in ("answer", "span", "quote")):
            continue
        cases.append({
            "question_id": base["question_id"], "transcript_id": base["transcript_id"],
            "question": base["question"],
            "control": {key: base[key] for key in ("answer", "span", "quote")},
            "candidate": {key: proposed[key] for key in ("answer", "span", "quote")},
        })
    return {
        "reference_spans_hidden": True, "cases": cases,
        "review_criteria": [
            "The answer is supported by this conversation, including polarity and temporal status.",
            "The citation preserves every requested attribute without importing unrelated claims.",
            "A valid alternative occurrence is not a semantic failure just because the annotation differs.",
            "Do not select a shorter or later citation solely because it would match a reference interval.",
        ],
    }


def run_variants(grouped, transcripts, frozen, tids, client, variants, output, budget_s):
    if not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("The explicit offline conversation budget must be positive.")
    if not variants or variants[0] != "base" or len(variants) != len(set(variants)) or not set(variants).issubset(ARMS):
        raise ValueError("Use the control first and at most the two preregistered prompt candidates.")
    if os.environ.get("MEDAPP_SPAN_CALIBRATION"):
        raise ValueError("The prompt probe applies its fixed correction once; inherited calibration is prohibited.")
    results = {}
    for variant in variants:
        recorded = RecordingClient(client)
        records, requests = [], []
        for tid in tids:
            group, transcript = grouped[tid], transcripts[tid]
            examples = reformat_frozen_demonstrations(frozen[tid]["few_shot"], variant)
            answerer = LLMAnswerer(client=recorded, few_shot=examples, variant=variant)
            recorded.calls.clear()
            began = time.monotonic()
            answers = answerer.answer_all(
                [row["question"] for row in group], transcript,
                deadline=began + budget_s, return_info=True,
            )
            elapsed = time.monotonic() - began
            current = records_for(group, answers, transcript["duration"])
            records.extend(current)
            requests.append({
                "transcript_id": tid, "latency_s": elapsed, "calls": list(recorded.calls),
                "generation": client.last_generation,
                "demonstration_tids": frozen[tid]["demonstration_tids"],
            })
            write_json(output / f"{variant}_questions.json", records)
            write_json(output / f"{variant}_conversations.json", requests)
            print(json.dumps({"variant": variant, "transcript_id": tid, "generation_s": elapsed}), flush=True)
        expected_ids = {row["question_id"] for tid in tids for row in grouped[tid]}
        if len(records) != len(expected_ids) or {row["question_id"] for row in records} != expected_ids:
            raise ValueError("The prompt comparison does not have complete unique question coverage.")
        failed = sum(row["decided_by"] not in {"yes", "no"} for row in records)
        results[variant] = {
            "records": records, "score": exact_score(records),
            "raw_score": exact_score([{**row, "span": row["raw_span"]} for row in records]),
            "failed_questions": failed, "max_component_s": max(request["latency_s"] for request in requests),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokenization", choices=("legacy", "template"), default="legacy")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--phase", choices=("feasibility", "pilot", "development", "confirmation"), default="feasibility")
    parser.add_argument("--variants", nargs="+", choices=ARMS, default=["base", "v1", "v1_claim"])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--conversation-budget-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("results/prompt_probe"))
    args = parser.parse_args()
    if os.name != "posix" or not os.environ.get("SLURM_JOB_ID") or not Path("run_request.json").is_file():
        parser.error("Run model probes in an isolated IDUN snapshot.")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use an immutable model revision.")
    if args.max_new_tokens < 1 or not math.isfinite(args.conversation_budget_s) or args.conversation_budget_s <= 0:
        parser.error("Token and offline time budgets must be positive.")
    if args.device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("CPU quality probes require an isolated CPU allocation.")
    if DEVICE not in ("auto", args.device):
        parser.error("The client device override disagrees with the declared probe device.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh prompt-probe output directory.")
    rows, requests, transcripts, _ = load_inputs(args.baseline)
    manifest_path = args.round / "round.json"
    manifest = json.loads(manifest_path.read_text())
    frozen = json.loads((args.round / "frozen_demonstrations.json").read_text())
    grouped = validate_inputs(rows, requests, transcripts, manifest, frozen)
    tids = selected_tids(manifest, args.phase)
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if threads < 1:
        raise ValueError("Allocated CPU thread count must be positive.")
    runtime = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")}
    signature = {
        "model": args.model, "revision": args.revision, "dtype": args.dtype, "device": args.device,
        "tokenization": args.tokenization, "enable_thinking": False if args.disable_thinking else None,
        "max_new_tokens": args.max_new_tokens,
        "conversation_budget_s": args.conversation_budget_s, "cpu_threads": threads,
        "runtime": runtime,
        "round_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    source_hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in PIPELINE_FILES
    }
    variants = args.variants
    if args.phase == "confirmation":
        if args.selection is None:
            parser.error("Confirmation requires a frozen --selection file.")
        selection = json.loads(args.selection.read_text())
        development_path = Path(selection["development_summary"])
        if hashlib.sha256(development_path.read_bytes()).hexdigest() != selection["development_summary_sha256"]:
            raise ValueError("The selected development result changed.")
        development = json.loads(development_path.read_text())
        variants = validate_confirmation(selection, development, manifest, signature, source_hashes)
    elif args.selection is not None:
        parser.error("--selection is only valid for confirmation.")
    import resource
    import torch

    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("GPU probes require exactly their allocated device.")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    report = {
        "complete": False, "phase": args.phase, "selected_tids": tids, "signature": signature,
        "pipeline_source_sha256": source_hashes,
        "definition_sha256": {variant: definition_hash(variant) for variant in variants},
        "diagnostic_only": True, "deployment_qualified": False, "serving_artifact_written": False,
        "runtime": runtime,
        "cpu_threads": threads, "variants": {},
        "limitations": [
            "Frozen-transcript generation timing is not uncached HTTP acceptance.",
            "CPU precision/device comparisons are not a matched GPU reproduction of the qualified endpoint.",
            "This repeatedly used corpus is not a virgin holdout; confirmation must not feed further prompt tuning.",
            "Semantic support must be judged independently of imperfect reference intervals.",
        ],
    }
    write_json(args.output / "summary.json", report)
    client = HFClient(
        model_name=args.model, revision=args.revision, dtype=args.dtype,
        max_new_tokens=args.max_new_tokens, legacy_special_tokens=args.tokenization == "legacy",
        num_beams=1, enable_thinking=False if args.disable_thinking else None,
    )
    started = time.monotonic()
    client.warm_up()
    report["load_and_warm_s"] = time.monotonic() - started
    if client._device != args.device or str(next(client._model.parameters()).dtype).removeprefix("torch.") != args.dtype:
        raise ValueError("The loaded model does not match the declared device/precision.")
    if args.phase == "feasibility":
        tid = tids[0]
        messages = build_l1_messages(
            transcripts[tid], [row["question"] for row in grouped[tid]], frozen[tid]["few_shot"], variant="base",
        )
        raw = client.generate(messages, max_new_tokens=64, deadline=time.monotonic() + args.conversation_budget_s)
        report.update({
            "complete": True, "feasibility_only": True, "generation": client.last_generation,
            "nonempty_completion": bool(raw.strip()),
        })
        write_json(args.output / "feasibility_completion.json", {"raw_output": raw})
    else:
        results = run_variants(
            grouped, transcripts, frozen, tids, client, variants, args.output, args.conversation_budget_s,
        )
        control = results["base"]["records"]
        qualified = [baseline_prediction(row) for tid in tids for row in grouped[tid]]
        report["qualified_reference"] = exact_score(qualified)
        report["strata_definition"] = "Fixed from the qualified reference predictions; not used in inference or per-question routing."
        for tid in tids:
            path = args.output / "semantic_transcripts" / f"{tid}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialize_transcript(transcripts[tid]), encoding="utf-8")
        for variant, value in results.items():
            report["variants"][variant] = {
                **{key: item for key, item in value.items() if key != "records"},
                "paired_same_runtime_control": paired_comparison(control, value["records"]),
                "paired_qualified_reference_runtime_not_matched": paired_comparison(qualified, value["records"]),
                "qualified_baseline_strata": localization_strata(qualified, control, value["records"]),
            }
            write_json(args.output / f"{variant}_semantic_changes.json", semantic_changes(control, value["records"]))
        report["complete"] = True
    report["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if args.phase == "feasibility":
        return 0 if report["nonempty_completion"] else 1
    return 0 if all(result["failed_questions"] == 0 for result in report["variants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
