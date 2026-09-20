"""Freeze a prompt-tuning round and reference-hidden semantic review inputs."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from answerers.llm_prompt import build_few_shot, build_l1_messages, select_few_shot_rows, serialize_transcript
from answerers.modernbert_data import grouped_folds
from audit_localization import exact_score, validate_qualified
from benchmark import write_json
from benchmark_alignment import baseline_prediction, load_inputs
from utils import temporal_iou


def prompt_hash(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def round_split(rows, requests):
    demos = {tid for request in requests for tid in request["demonstration_tids"]}
    tids = {row["transcript_id"] for row in rows}
    eligible = tids - demos
    if len(eligible) < 5:
        raise ValueError("Prompt tuning requires at least five demonstration-disjoint conversations.")
    folds = grouped_folds(eligible, 5, 13)
    development = set(folds[0]) | set(folds[1])
    confirmation = eligible - development
    return {
        "seed": 13, "folds": folds, "development_tids": sorted(development),
        "confirmation_tids": sorted(confirmation), "excluded_demonstration_tids": sorted(demos),
        "pilot_tids": folds[0][:3],
    }


def stratum(row):
    if row["label"] != 1:
        return row["question_type"]
    if not row["answer"]:
        return "missed_positive"
    value = temporal_iou(row["gold"], row["span"])
    return "disjoint" if value == 0 else "low_tiou" if value < 0.5 else "high_tiou" if value >= 0.8 else "mid_tiou"


def review_cases(rows, development_tids, per_stratum=3):
    selected = [baseline_prediction(row) for row in rows if row["transcript_id"] in set(development_tids)]
    grouped = defaultdict(list)
    for row in sorted(selected, key=lambda item: (item["transcript_id"], item["question_id"])):
        grouped[stratum(row)].append(row)
    cases = []
    for name in ("disjoint", "low_tiou", "high_tiou", "hard_negative", "off_topic", "missed_positive"):
        used = set()
        for row in grouped[name]:
            if row["transcript_id"] in used:
                continue
            cases.append({
                "question_id": row["question_id"], "transcript_id": row["transcript_id"], "stratum": name,
                "question": row["question"], "baseline_answer": row["answer"],
                "baseline_quote": row.get("quote"), "baseline_span": row["span"],
            })
            used.add(row["transcript_id"])
            if len(used) == per_stratum:
                break
    required = {"disjoint", "low_tiou", "high_tiou"}
    if not required.issubset({case["stratum"] for case in cases}):
        raise ValueError("The frozen development split does not cover all requested localization strata.")
    return cases, dict(Counter(stratum(row) for row in selected))


def frozen_demonstrations(demo_root, rows, requests, transcripts):
    with (demo_root / "data" / "question_train.csv").open(encoding="utf-8", newline="") as handle:
        demo_rows = list(csv.DictReader(handle))
    with (demo_root / "annotations" / "evidence.csv").open(encoding="utf-8", newline="") as handle:
        evidence = {row["question_id"]: row for row in csv.DictReader(handle)}
    by_tid = defaultdict(list)
    for row in demo_rows:
        by_tid[row["transcript_id"]].append(row)
    targets = defaultdict(list)
    for row in rows:
        targets[row["transcript_id"]].append(row)
    sources = sorted({tid for request in requests for tid in request["demonstration_tids"]})
    demo_transcripts, file_hashes = {}, {}
    for tid in sources:
        path = demo_root / "transcripts" / f"conversation_{tid}.dc5ba020.json"
        if not path.is_file():
            raise FileNotFoundError(f"The historical demonstration cache is missing: {path}")
        demo_transcripts[tid] = json.loads(path.read_text())
        file_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = {}
    for request in requests:
        tid = request["transcript_id"]
        selected = select_few_shot_rows(by_tid, demo_transcripts, evidence, tid)
        if [row["transcript_id"] for row in selected] != request["demonstration_tids"]:
            raise ValueError(f"Demonstration source identities changed: {tid}")
        examples = build_few_shot(by_tid, demo_transcripts, evidence, tid, variant="base")
        messages = build_l1_messages(
            transcripts[tid], [row["question"] for row in targets[tid]], examples, variant="base",
        )
        if len(request["calls"]) != 1 or prompt_hash(messages) != request["calls"][0]["prompt_sha256"]:
            raise ValueError(f"The strongest incumbent's recorded prompt does not reproduce: {tid}")
        frozen[tid] = {
            "few_shot": examples, "prompt_sha256": prompt_hash(messages),
            "question_ids": [row["question_id"] for row in targets[tid]],
            "demonstration_tids": request["demonstration_tids"],
        }
    return frozen, file_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--qualified", type=Path, required=True)
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/prompt_round"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh prompt-round output directory.")
    rows, requests, transcripts, _ = load_inputs(args.baseline)
    qualified = json.loads((args.qualified / "questions.json").read_text())
    qualified_summary = json.loads((args.qualified / "summary.json").read_text())
    corrected = [baseline_prediction(row) for row in rows]
    validate_qualified(corrected, qualified, qualified_summary)
    split = round_split(rows, requests)
    frozen, demo_hashes = frozen_demonstrations(args.demo_root, rows, requests, transcripts)
    cases, counts = review_cases(rows, split["development_tids"])
    manifest = {
        "version": 1, **split, "control": exact_score(corrected),
        "development": exact_score([row for row in corrected if row["transcript_id"] in split["development_tids"]]),
        "development_strata": counts, "review_question_ids": [case["question_id"] for case in cases],
        "feasibility_tid": min(split["development_tids"], key=lambda tid: (len(transcripts[tid]["words"]), tid)),
        "all_recorded_control_prompt_hashes_reproduced": True,
        "demonstration_cache": "historical large-v3 dc5ba020; target transcripts remain frozen turbo",
        "max_new_prompt_candidates": 2,
        "confirmation_is_virgin_holdout": False,
        "confirmation_policy": "Freeze the candidate from development before confirmation; no retuning from this round's confirmation.",
        "semantic_policy": "Judge the chosen evidence against the conversation independently of the supplied span; never imitate unrelated labels.",
        "source_sha256": {
            **demo_hashes,
            **{
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    args.baseline / "base_legacy_questions.json", args.baseline / "base_legacy_conversations.json",
                    args.qualified / "questions.json", args.qualified / "summary.json",
                    args.demo_root / "data" / "question_train.csv", args.demo_root / "annotations" / "evidence.csv",
                )
            },
        },
    }
    write_json(args.output / "round.json", manifest)
    write_json(args.output / "frozen_demonstrations.json", frozen)
    write_json(args.output / "review_cases.json", {
        "reference_spans_hidden": True, "cases": cases,
        "review_fields": [
            "claim_supported", "appropriate_speaker_and_status", "necessary_qualifiers_present",
            "unnecessary_other_claims", "alternative_valid_occurrence", "recommended_general_rule",
        ],
    })
    for tid in sorted({case["transcript_id"] for case in cases}):
        destination = args.output / "review_transcripts" / f"{tid}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialize_transcript(transcripts[tid]), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_sha256"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
