"""Bounded source-unit geometry probe; reference-assisted ceilings are not scores."""

import argparse
import hashlib
import json
import logging
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from answerers.evidence_units import (
    SOURCE_RECIPE, build_evidence_units, incumbent_unit, shortlist_evidence,
)
from answerers.span_targets import optimal_quote_target
from answerers.source_ranker import InverseQuestionScorer, MODEL, POLICIES, RANK_RECIPE, REVISION, select_source
from audit_localization import exact_score
from benchmark import paired_comparison, write_json
from benchmark_alignment import baseline_prediction, load_inputs
from train_quote_oof import validate_coverage
from utils import temporal_iou


logger = logging.getLogger(__name__)


def validate_audit_replay(rows, audited, summary, recipe):
    predictions = [baseline_prediction(row) for row in rows]
    validate_coverage(audited, predictions)
    recorded = {row["question_id"]: row for row in audited}
    if any(row["span"] != recorded[row["question_id"]]["span"] for row in predictions):
        raise ValueError("The annotation audit used different incumbent spans.")
    actual = exact_score(predictions)
    if any(not math.isclose(actual[key], summary["incumbent"][key], rel_tol=0, abs_tol=1e-12)
           for key in ("score", "mean_tiou", "accuracy", "questions", "positives")):
        raise ValueError("The annotation audit's exact incumbent does not reproduce.")
    if (
        summary.get("diagnostic_only") is not True
        or recipe.get("version") != 1
        or recipe.get("planned_source_shortlist") != SOURCE_RECIPE["shortlist_units"]
        or recipe.get("offsets_s") != SOURCE_RECIPE["offsets_s"]
    ):
        raise ValueError("Source candidate budgets or timing differ from the frozen audit recipe.")
    return predictions


def build_runtime_case(row, transcript, inventory):
    baseline = baseline_prediction(row)
    incumbent = (
        incumbent_unit(transcript["words"], row["duration"], baseline["span"], row["word_range"])
        if row["answer"] else None
    )
    selected = shortlist_evidence(
        row["question"], inventory, row["duration"], incumbent=incumbent,
        top_k=SOURCE_RECIPE["shortlist_units"],
    )
    return {
        "question_id": row["question_id"], "transcript_id": row["transcript_id"],
        "question": row["question"], "baseline_answer": row["answer"],
        "baseline_span": baseline["span"],
        "candidates": [
            {**asdict(unit), "resolved_span": unit.resolved_span(row["duration"])} for unit in selected
        ],
    }, incumbent, selected


def best_unit(units, gold, duration):
    if not units:
        return {"tiou": 0.0, "span": None, "word_range": None, "families": [], "unit_index": None}
    ranked = [(unit, unit.resolved_span(duration)) for unit in units]
    unit, span = max(ranked, key=lambda item: (
        temporal_iou(gold, item[1]), -(item[1][1] - item[1][0]), -item[1][0],
    ))
    return {
        "tiou": temporal_iou(gold, span), "span": span,
        "word_range": list(unit.word_range()), "families": list(unit.families),
        "unit_index": unit.index, "quote": unit.text,
    }


def contained_subrange_oracle(units, gold, words, duration):
    best = best_unit(units, gold, duration)
    for unit in units:
        target = optimal_quote_target(
            words[unit.first_word:unit.last_word + 1], gold, duration,
            unique=False, offsets=(0.2, 0.0),
        )
        if target is not None and target["tiou"] > best["tiou"]:
            best = {
                "tiou": target["tiou"], "span": target["span"],
                "word_range": [unit.first_word + target["first_word"], unit.first_word + target["last_word"]],
                "quote": target["quote"], "families": list(unit.families), "unit_index": unit.index,
            }
    return {**best, "requires_unimplemented_subrange_selection": True}


def probe_geometry(rows, transcripts):
    inventories = {
        tid: build_evidence_units(transcript["words"], transcript["duration"])
        for tid, transcript in transcripts.items()
    }
    runtime, diagnostics = [], []
    for row in rows:
        tid, duration = row["transcript_id"], row["duration"]
        entry, incumbent, selected = build_runtime_case(row, transcripts[tid], inventories[tid])
        runtime.append(entry)
        pool = [*inventories[tid], *([incumbent] if incumbent is not None else [])]
        diagnostic = {
            "question_id": row["question_id"], "transcript_id": tid,
            "label": row["label"], "gold": row["gold"], "baseline_answer": row["answer"],
            "pool_units": len(pool), "shortlist_units": len(selected), "oracles": None,
        }
        if row["label"] == 1:
            diagnostic["oracles"] = {
                "all_units": best_unit(pool, row["gold"], duration),
                "shortlisted_units": best_unit(selected, row["gold"], duration),
                **{
                    family: best_unit([unit for unit in pool if family in unit.families], row["gold"], duration)
                    for family in ("clause", "sentence", "episode", "incumbent")
                },
                "subranges_inside_shortlisted_units": contained_subrange_oracle(
                    selected, row["gold"], transcripts[tid]["words"], duration,
                ),
            }
        diagnostics.append(diagnostic)
    return runtime, diagnostics, inventories


def geometry_summary(diagnostics):
    positive = [row for row in diagnostics if row["label"] == 1]
    if not positive:
        raise ValueError("Candidate coverage requires reference positives.")
    return {
        "questions": len(diagnostics), "positives": len(positive),
        "shortlist_mean": statistics.fmean(row["shortlist_units"] for row in diagnostics),
        "shortlist_max": max(row["shortlist_units"] for row in diagnostics),
        "candidate_oracles": {
            name: {
                "all_positive_mean_tiou": statistics.fmean(row["oracles"][name]["tiou"] for row in positive),
                "frozen_decision_mean_tiou": statistics.fmean(
                    row["oracles"][name]["tiou"] if row["baseline_answer"] else 0.0 for row in positive
                ),
                "at_least_0_8": sum(row["oracles"][name]["tiou"] >= 0.8 for row in positive),
                "at_least_0_95": sum(row["oracles"][name]["tiou"] >= 0.95 for row in positive),
                "positive_denominator": len(positive),
            }
            for name in positive[0]["oracles"]
        },
    }


def ranked_prediction(row, units, scores, policy):
    if not row["answer"] or len(units) != len(scores):
        raise ValueError("Ranking requires a retained yes decision and one likelihood row per candidate.")
    selected = select_source(scores, policy)
    unit = units[selected]
    return {
        **row, "span": unit.resolved_span(row["duration"]), "word_range": list(unit.word_range()),
        "quote": unit.text, "baseline_span": baseline_prediction(row)["span"],
        "baseline_word_range": row["word_range"], "source_reason": "selected",
        "source_unit_index": unit.index, "source_families": list(unit.families),
        "source_timing": unit.timing, "source_policy": policy,
    }


def verify_model_cache(path):
    from huggingface_hub import snapshot_download

    manifest = json.loads(path.read_text())
    if (
        manifest.get("model") != MODEL or manifest.get("revision") != REVISION
        or manifest.get("complete") is not True or manifest.get("inference_network_access") is not False
    ):
        raise ValueError("Source scoring requires the complete pinned offline model-cache manifest.")
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    if snapshot.resolve() != Path(manifest["path"]).resolve():
        raise ValueError("The source ranker would load a different cached snapshot.")
    for entry in manifest["files"]:
        target = snapshot / entry["path"]
        if not target.is_file() or target.stat().st_size != entry["bytes"]:
            raise ValueError(f"Source model cache changed: {entry['path']}")
    with (snapshot / "model.safetensors").open("rb") as handle:
        weights_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "cache_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "weights_sha256": weights_sha256,
        "model": MODEL, "revision": REVISION, "path": str(snapshot),
    }


def run_ranking(rows, transcripts, excluded, output, *, smoke=False):
    import resource

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run the frozen source-ranker experiment in an isolated CPU allocation.")
    selected_tids = (
        sorted(transcripts, key=lambda tid: (-transcripts[tid]["duration"], tid))[:2]
        if smoke else sorted(transcripts)
    )
    expected = [row for row in rows if row["transcript_id"] in selected_tids]
    baseline = [baseline_prediction(row) for row in expected]
    predictions = {policy: [] for policy in POLICIES}
    traces, timings, failures = [], [], []
    scorer = InverseQuestionScorer()
    load_started = time.monotonic()
    scorer.load()
    load_s = time.monotonic() - load_started
    for tid in selected_tids:
        group = [row for row in expected if row["transcript_id"] == tid]
        chosen = [row for row in group if row["answer"]]
        if smoke:
            chosen = sorted(chosen, key=lambda row: (-len(row["question"]), row["question_id"]))[:2]
        started = time.monotonic()
        inventory = build_evidence_units(transcripts[tid]["words"], transcripts[tid]["duration"])
        selections, cases = {}, []
        for row in chosen:
            _, _, candidates = build_runtime_case(row, transcripts[tid], inventory)
            if smoke:
                candidates = [candidates[0], *sorted(
                    candidates[1:], key=lambda unit: (-(unit.context_last_word - unit.context_first_word), unit.index),
                )[:7]]
            selections[row["question_id"]] = candidates
            cases.append((row["question"], candidates))
        likelihoods, error = [], None
        if cases:
            try:
                likelihoods = scorer.score(cases, transcripts[tid]["words"], check_direct=smoke)
            except (RuntimeError, ValueError, OSError, MemoryError) as exc:
                logger.exception("Source scoring failed for %s; retaining every incumbent answer/span.", tid)
                error = f"{type(exc).__name__}: {exc}"
                failures.append({"transcript_id": tid, "error": error})
        scores_by_id = {
            row["question_id"]: scores for row, scores in zip(chosen, likelihoods)
        } if error is None else {}
        if error is None and len(scores_by_id) != len(chosen):
            raise ValueError("The source ranker omitted a requested question.")
        for row in group:
            scores = scores_by_id.get(row["question_id"])
            for policy in POLICIES:
                predictions[policy].append(
                    ranked_prediction(row, selections[row["question_id"]], scores, policy) if scores is not None else
                    {**baseline_prediction(row), "source_reason": (
                        "conversation_error_keep" if error is not None and row["answer"]
                        else "smoke_unscored" if row["answer"] else "base_no"
                    )}
                )
            if scores is not None:
                traces.append({
                    "question_id": row["question_id"], "transcript_id": tid,
                    "candidates": [
                        {**asdict(unit), "resolved_span": unit.resolved_span(row["duration"]), "likelihoods": value}
                        for unit, value in zip(selections[row["question_id"]], scores)
                    ],
                })
        timings.append({
            "transcript_id": tid, "elapsed_s": time.monotonic() - started,
            "error": error, "scoring": scorer.last_metrics if cases and error is None else None,
        })
        print(json.dumps({"source_scoring": timings[-1]}), flush=True)
    for policy, records in predictions.items():
        validate_coverage(records, expected)
        write_json(output / f"{policy}_questions.json", records)
    report = {
        "diagnostic_only": True, "mode": "rank", "smoke_only": smoke, "complete_corpus": not smoke,
        "serving_artifact_written": False, "deployment_qualified": False,
        "trained": False, "primary_policy": RANK_RECIPE["primary_policy"],
        "model": MODEL, "revision": REVISION, "device": "cpu", "load_s": load_s,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "scored_question_count": len(traces), "conversation_failures": failures,
        "likelihood_feasibility_passed": bool(traces) and not failures,
        "max_component_s": max(entry["elapsed_s"] for entry in timings),
        "excluded_demonstration_tids": excluded, "comparisons": {},
        "limitations": [
            "The current whole-unit shortlist does not pass the near-perfect coverage gate.",
            "This tests pretrained inverse-question signal, not a fitted selector or acoustic boundaries.",
            "A smoke result is feasibility only; CPU component timing is not uncached HTTP acceptance.",
            "All decisions and official reference labels remain unchanged, including suspect greeting references.",
            "Repeated development on this corpus limits generalization claims even for a positive paired interval.",
        ],
    }
    if not smoke:
        for policy, records in predictions.items():
            safe = [row for row in records if row["transcript_id"] not in excluded]
            safe_base = [row for row in baseline if row["transcript_id"] not in excluded]
            report["comparisons"][policy] = {
                "all_questions": {
                    "baseline": exact_score(baseline), "candidate": exact_score(records),
                    "paired": paired_comparison(baseline, records),
                },
                "demo_disjoint": {
                    "baseline": exact_score(safe_base), "candidate": exact_score(safe),
                    "paired": paired_comparison(safe_base, safe),
                },
            }
    write_json(output / "likelihoods.json", traces)
    write_json(output / "timings.json", timings)
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["likelihood_feasibility_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--mode", choices=("geometry", "rank"), default="geometry")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/source_localization"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.mode == "rank" and args.cache_manifest is None:
        parser.error("Rank mode requires --cache-manifest for the pinned offline model.")
    if args.mode == "geometry" and (args.cache_manifest is not None or args.smoke):
        parser.error("Model-cache and smoke controls only apply to rank mode.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh source-localization output directory.")
    rows, requests, transcripts, _ = load_inputs(args.baseline)
    audited = json.loads((args.audit / "questions.json").read_text())
    summary = json.loads((args.audit / "summary.json").read_text())
    recipe = json.loads((args.audit / "frozen_recipe.json").read_text())
    predictions = validate_audit_replay(rows, audited, summary, recipe)
    for name in ("base_legacy_questions.json", "base_legacy_conversations.json"):
        digests = [digest for path, digest in recipe["source_sha256"].items() if Path(path).name == name]
        actual = hashlib.sha256((args.baseline / name).read_bytes()).hexdigest()
        if digests != [actual]:
            raise ValueError(f"Frozen annotation-audit inputs changed: {name}")
    excluded = sorted({tid for request in requests for tid in request["demonstration_tids"]})
    if excluded != recipe["excluded_demonstration_tids"]:
        raise ValueError("Demonstration-source exclusions changed after the annotation audit.")
    write_json(args.output / "frozen_recipe.json", {
        **SOURCE_RECIPE, "job_id": os.environ.get("SLURM_JOB_ID"),
        "audit_recipe_sha256": hashlib.sha256((args.audit / "frozen_recipe.json").read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256((args.baseline / "base_legacy_questions.json").read_bytes()).hexdigest(),
        "outer_folds": recipe["outer_folds"], "excluded_demonstration_tids": excluded,
        "mode": args.mode, "serving_artifact_written": False,
        "source_ranker": RANK_RECIPE if args.mode == "rank" else None,
        "smoke_only": args.smoke,
    })
    if args.mode == "rank":
        write_json(args.output / "model_provenance.json", verify_model_cache(args.cache_manifest))
        return run_ranking(rows, transcripts, excluded, args.output, smoke=args.smoke)
    runtime, diagnostics, inventories = probe_geometry(rows, transcripts)
    report = {
        "diagnostic_only": True, "mode": "geometry_only", "serving_artifact_written": False,
        "incumbent": exact_score(predictions),
        "all_questions": geometry_summary(diagnostics),
        "demo_disjoint": geometry_summary([row for row in diagnostics if row["transcript_id"] not in excluded]),
        "excluded_demonstration_tids": excluded,
        "inventory_counts": {tid: len(units) for tid, units in inventories.items()},
        "limitations": [
            "Reference-assisted unit coverage is not a model score or an achieved improvement.",
            "The lexical shortlist seeds a source-ranker experiment; it never changes a boolean.",
            "Episode candidates are contiguous sentence sequences, not verified clinical actions.",
            "Interpretation context is not added to a citation when measuring its IoU.",
            "All reference positives remain in frozen-decision denominators, including missed positives.",
            "The contained-subrange oracle diagnoses source-versus-extent loss; no subrange selector is implemented.",
        ],
    }
    write_json(args.output / "runtime_candidates.json", runtime)
    write_json(args.output / "diagnostics.json", diagnostics)
    write_json(args.output / "summary.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "inventory_counts"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
