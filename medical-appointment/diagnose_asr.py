"""Separate precision-related timing changes from changed evidence selection."""

import argparse
import json
from pathlib import Path

from answerers.align import align_quote_matches
from answerers.boundaries import adjusted_span
from benchmark import paired_comparison, write_json
from calibrate_spans import apply_offsets, cross_validate
from llm_probe import score_records


def load_run(directory):
    rows = json.loads((directory / "base_legacy_questions.json").read_text())
    requests = json.loads((directory / "base_legacy_conversations.json").read_text())
    words = {
        request["transcript_id"]: json.loads(
            (directory / "transcripts" / f"{request['transcript_id']}.json").read_text()
        )["words"]
        for request in requests
    }
    return rows, requests, words


def crossover(rows, target_words):
    """Retain decisions and quote occurrence; swap timing only when anchored."""
    results = []
    aligned_count = rejected_count = 0
    for row in rows:
        source_span = row["span"]
        selected = source_span
        if row["answer"] and source_span is not None:
            matches = align_quote_matches(target_words[row["transcript_id"]], row.get("quote") or "")
            if matches:
                nearest = min(
                    matches,
                    key=lambda match: abs(match[0] - source_span[0]) + abs(match[1] - source_span[1]),
                )
                if max(abs(nearest[0] - source_span[0]), abs(nearest[1] - source_span[1])) <= 1.0:
                    selected = list(nearest[:2])
                    aligned_count += 1
                else:
                    rejected_count += 1
            else:
                rejected_count += 1
        results.append({
            **row,
            "span": adjusted_span(selected, (0.2, 0.0), row.get("duration"))
            if row["answer"] else None,
        })
    return results, {"aligned": aligned_count, "retained_source": rejected_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/precision_diagnosis"))
    args = parser.parse_args()
    baseline, base_requests, base_words = load_run(args.baseline)
    candidate, cand_requests, cand_words = load_run(args.candidate)
    if {r["question_id"] for r in baseline} != {r["question_id"] for r in candidate}:
        raise ValueError("Precision runs cover different questions.")
    base_audio = {r["transcript_id"]: r["audio_sha256"] for r in base_requests}
    cand_audio = {r["transcript_id"]: r["audio_sha256"] for r in cand_requests}
    if base_audio != cand_audio:
        raise ValueError("Precision runs use different audio.")
    excluded = {
        tid for request in [*base_requests, *cand_requests]
        for tid in request["demonstration_tids"]
    }
    baseline_corrected = apply_offsets(baseline, (0.2, 0.0))
    safe_baseline = [r for r in baseline_corrected if r["transcript_id"] not in excluded]
    equal_text = sum(
        [w["word"] for w in base_words[tid]] == [w["word"] for w in cand_words[tid]]
        for tid in base_words
    )
    equal_times = sum(
        [(w["start"], w["end"]) for w in base_words[tid]]
        == [(w["start"], w["end"]) for w in cand_words[tid]]
        for tid in base_words
    )
    report = {
        "identical_word_text_conversations": equal_text,
        "identical_word_times_conversations": equal_times,
        "conversations": len(base_words),
        "excluded_demonstration_tids": sorted(excluded),
        "baseline": score_records(safe_baseline),
        "crossovers": {},
        "candidate_calibration": {},
    }
    for name, rows, words in (
        ("int8_quotes_fp16_times", baseline, cand_words),
        ("fp16_quotes_int8_times", candidate, base_words),
    ):
        crossed, coverage = crossover(rows, words)
        safe = [r for r in crossed if r["transcript_id"] not in excluded]
        result = {
            "score": score_records(safe), "paired": paired_comparison(safe_baseline, safe),
            "coverage": coverage,
        }
        report["crossovers"][name] = result
        write_json(args.output / f"{name}.json", crossed)
        print(json.dumps({"method": name, **result}), flush=True)
    for seed in (13, 37):
        _, predicted, folds = cross_validate(candidate, excluded, n_folds=5, seed=seed)
        result = {
            "score": score_records(predicted),
            "paired": paired_comparison(safe_baseline, predicted, seed=seed),
            "offsets": [fold["offsets"] for fold in folds],
        }
        report["candidate_calibration"][str(seed)] = result
        print(json.dumps({"calibration_seed": seed, **result}), flush=True)
    write_json(args.output / "summary.json", report)
    print(f"Identical words: {equal_text}/39; identical word times: {equal_times}/39.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
