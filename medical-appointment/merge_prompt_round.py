"""Combine every fixed execution shard without selecting conversations or outputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from answerers.base import normalize_answer
from answerers.boundaries import adjusted_span
from answerers.llm_prompt import serialize_transcript
from audit_localization import exact_score
from benchmark import paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs
from probe_prompt_round import (
    ARMS, ORDER_CONTROL_EXTENSION, definition_hash, localization_strata,
    matched_analysis_rows, pipeline_hashes, selected_tids, semantic_changes, validate_inputs,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_records(reference, records):
    matched_analysis_rows(reference, reference, records)
    original = {row["question_id"]: row for row in reference}
    for row in records:
        if (
            not isinstance(row["answer"], bool) or row["prediction"] != int(row["answer"])
            or row["duration"] != original[row["question_id"]]["duration"]
        ):
            raise ValueError("A prompt shard changed prediction types or frozen audio duration.")
        valid, checked = normalize_answer(
            (row["answer"], row["span"]), duration=row["duration"], context=row["question_id"],
        )
        expected_span = adjusted_span(row["raw_span"], (0.2, 0.0), row["duration"]) if row["answer"] else None
        if (
            valid != row["answer"] or (list(checked) if checked is not None else None) != row["span"]
            or row["span"] != expected_span
        ):
            raise ValueError("A prompt shard violated the span contract or applied calibration inconsistently.")


def merge_results(rows, manifest, part_dirs, phase, source_hashes, round_hash):
    if phase not in ("development", "confirmation") or len(part_dirs) < 2:
        raise ValueError("Merge every shard of a development or confirmation phase.")
    summaries = [read_json(path / "summary.json") for path in part_dirs]
    template = summaries[0]
    variants = list(template["variants"])
    phase_tids = selected_tids(manifest, phase)
    count = len(part_dirs)
    if (
        not variants or "base" not in variants or not set(variants).issubset(ARMS)
        or ("claim" in variants and set(variants) != {"base", "claim"})
        or template["signature"].get("round_sha256") != round_hash
    ):
        raise ValueError("The prompt shards do not describe the frozen round and declared arms.")
    indices, combined, calls, bindings = set(), {variant: [] for variant in variants}, {variant: [] for variant in variants}, []
    for directory, summary in zip(part_dirs, summaries):
        shard = summary.get("execution_shard", {})
        index = shard.get("index")
        if (
            type(index) is not int or index in indices or not 0 <= index < count
            or shard.get("count") != count
            or summary.get("complete") is not True or summary.get("full_phase") is not False
            or summary.get("phase") != phase or summary.get("phase_tids") != phase_tids
            or summary.get("selected_tids") != selected_tids(manifest, phase, index, count)
            or summary.get("signature") != template["signature"]
            or summary.get("pipeline_source_sha256") != source_hashes
            or summary.get("definition_sha256") != {variant: definition_hash(variant) for variant in variants}
            or set(summary.get("variants", {})) != set(variants)
            or summary.get("protocol_extension") != (ORDER_CONTROL_EXTENSION if "claim" in variants else None)
        ):
            raise ValueError("A prompt shard is incomplete, duplicated, mismatched, or not from the frozen partition.")
        indices.add(index)
        reference = [row for row in rows if row["transcript_id"] in summary["selected_tids"]]
        binding = {
            "directory": str(directory), "index": index,
            "summary_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
            "files_sha256": {},
        }
        for variant in variants:
            records_path = directory / f"{variant}_questions.json"
            requests_path = directory / f"{variant}_conversations.json"
            records, requests = read_json(records_path), read_json(requests_path)
            validate_records(reference, records)
            actual = exact_score(records)
            if (
                any(not math.isclose(actual[key], summary["variants"][variant]["score"][key], rel_tol=0, abs_tol=1e-12)
                    for key in ("questions", "positives", "accuracy", "score", "mean_tiou"))
                or sum(row["decided_by"] not in {"yes", "no"} for row in records) != summary["variants"][variant]["failed_questions"]
                or len(requests) != len(summary["selected_tids"])
                or {request["transcript_id"] for request in requests} != set(summary["selected_tids"])
                or any(not math.isfinite(request["latency_s"]) or request["latency_s"] < 0 for request in requests)
            ):
                raise ValueError("A prompt shard's saved predictions, failures, or requests do not reproduce.")
            combined[variant].extend(records)
            calls[variant].extend(requests)
            for path in (records_path, requests_path):
                binding["files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        bindings.append(binding)
    if indices != set(range(count)):
        raise ValueError("A prompt execution shard is missing.")
    expected = [row for tid in phase_tids for row in rows if row["transcript_id"] == tid]
    qualified = [baseline_prediction(row) for row in expected]
    for variant, records in combined.items():
        validate_records(expected, records)
        by_id = {row["question_id"]: row for row in records}
        combined[variant] = [by_id[row["question_id"]] for row in expected]
        requests = {request["transcript_id"]: request for request in calls[variant]}
        calls[variant] = [requests[tid] for tid in phase_tids]
    control = combined["base"]
    report = {
        "complete": True, "full_phase": True, "phase": phase, "selected_tids": phase_tids,
        "phase_tids": phase_tids, "signature": template["signature"],
        "pipeline_source_sha256": source_hashes, "definition_sha256": template["definition_sha256"],
        "protocol_extension": template.get("protocol_extension"),
        "diagnostic_only": True, "deployment_qualified": False, "serving_artifact_written": False,
        "merged_execution_shards": bindings,
        "runtime": template["runtime"], "cpu_threads": template["cpu_threads"],
        "qualified_reference": exact_score(qualified),
        "strata_definition": template["strata_definition"], "variants": {},
        "limitations": [
            *template["limitations"],
            "Resource shards partition execution, not the development/confirmation groups or scoring denominator.",
            "All predefined shards and every failed output are included; no best-of-runs selection is permitted.",
        ],
    }
    for variant, records in combined.items():
        report["variants"][variant] = {
            "score": exact_score(records),
            "raw_score": exact_score([{**row, "span": row["raw_span"]} for row in records]),
            "failed_questions": sum(row["decided_by"] not in {"yes", "no"} for row in records),
            "max_component_s": max(request["latency_s"] for request in calls[variant]),
            "paired_same_runtime_control": paired_comparison(control, records),
            "paired_qualified_reference_runtime_not_matched": paired_comparison(qualified, records),
            "qualified_baseline_strata": localization_strata(qualified, control, records),
        }
    return report, combined, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--round", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/prompt_phase"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh merged prompt-phase output directory.")
    rows, requests, transcripts, _ = load_inputs(args.baseline)
    manifest_path = args.round / "round.json"
    manifest = read_json(manifest_path)
    validate_inputs(rows, requests, transcripts, manifest, read_json(args.round / "frozen_demonstrations.json"))
    report, records, calls = merge_results(
        rows, manifest, args.parts, args.phase, pipeline_hashes(),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    for variant, values in records.items():
        write_json(args.output / f"{variant}_questions.json", values)
        write_json(args.output / f"{variant}_conversations.json", calls[variant])
        write_json(args.output / f"{variant}_semantic_changes.json", semantic_changes(records["base"], values))
    for tid in report["selected_tids"]:
        path = args.output / "semantic_transcripts" / f"{tid}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialize_transcript(transcripts[tid]), encoding="utf-8")
    write_json(args.output / "summary.json", report)
    print(json.dumps({
        "complete": True, "phase": report["phase"], "selected_tids": report["selected_tids"],
        "variants": {variant: {key: value for key, value in result.items() if key != "qualified_baseline_strata"}
                     for variant, result in report["variants"].items()},
    }, indent=2), flush=True)
    return 0 if all(result["failed_questions"] == 0 for result in report["variants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
