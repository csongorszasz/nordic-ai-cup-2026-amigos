"""CPU-only anchored span ranking with nested conversation-grouped validation."""

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from answer import _STOPWORDS, _content_tokens
from answerers.boundaries import adjusted_span
from answerers.modernbert_data import grouped_folds, load_rows
from answerers.passages import overlap_word_range
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from utils import gold_evidence, temporal_iou

ALPHAS = (0.01, 0.1, 1.0)
MARGINS = (0.0, 0.01, 0.03, 0.05)
FEATURES = (
    "log_duration", "length_ratio", "length_ratio_squared", "log_words",
    "question_overlap", "number_overlap", "first_stopword", "last_stopword",
    "preceding_punctuation", "ending_punctuation", "segment_start", "segment_end",
    "preceding_pause", "following_pause", "start_delta", "end_delta",
    "absolute_start_delta", "absolute_end_delta", "start_delta_squared",
    "end_delta_squared", "relative_start", "relative_end", "baseline_overlap",
    "word_probability",
)


@dataclass
class Case:
    row: dict
    spans: list
    features: np.ndarray
    targets: np.ndarray
    gram: np.ndarray
    cross: np.ndarray


def candidates(question, words, baseline, duration, context_words=4):
    """Runtime-only inputs: no labels or reference evidence enter this function."""
    if baseline is None:
        return [None], np.zeros((1, len(FEATURES)))
    bounds = overlap_word_range(words, *baseline)
    if bounds is None:
        raise ValueError("Baseline evidence overlaps no ASR words.")
    lo = max(0, bounds[0] - context_words)
    hi = min(len(words) - 1, bounds[1] + context_words)
    base_duration = max(0.2, baseline[1] - baseline[0])
    qtokens = _content_tokens(question)
    numbers = set(re.findall(r"\d+(?:\.\d+)?", question))
    clean = [word["word"].strip() for word in words]
    normalized = [re.sub(r"^\W+|\W+$", "", word.lower()) for word in clean]
    overlap = [len(_content_tokens(word) & qtokens) for word in clean]
    numeric = [len(set(re.findall(r"\d+(?:\.\d+)?", word)) & numbers) for word in clean]
    probabilities = [float(word.get("probability", 1.0)) for word in words]
    prefix_overlap = np.concatenate(([0.0], np.cumsum(overlap)))
    prefix_numeric = np.concatenate(([0.0], np.cumsum(numeric)))
    prefix_probability = np.concatenate(([0.0], np.cumsum(probabilities)))

    def features(span, first, last):
        length = span[1] - span[0]
        count = last - first + 1
        ratio = min(4.0, length / base_duration)
        ds = max(-4.0, min(4.0, (span[0] - baseline[0]) / base_duration))
        de = max(-4.0, min(4.0, (span[1] - baseline[1]) / base_duration))
        prev = first - 1
        nxt = last + 1
        return [
            math.log1p(length), ratio, ratio * ratio, math.log1p(count),
            min(1.0, (prefix_overlap[last + 1] - prefix_overlap[first]) / max(1, len(qtokens))),
            min(1.0, (prefix_numeric[last + 1] - prefix_numeric[first]) / max(1, len(numbers))),
            float(normalized[first] in _STOPWORDS), float(normalized[last] in _STOPWORDS),
            float(prev < 0 or clean[prev].endswith((".", "?", "!", ",", ";", ":"))),
            float(clean[last].endswith((".", "?", "!", ",", ";", ":"))),
            float(prev < 0 or words[first].get("seg_idx") != words[prev].get("seg_idx")),
            float(nxt >= len(words) or words[last].get("seg_idx") != words[nxt].get("seg_idx")),
            min(2.0, max(0.0, words[first]["start"] - words[prev]["end"])) if prev >= 0 else 0.0,
            min(2.0, max(0.0, words[nxt]["start"] - words[last]["end"])) if nxt < len(words) else 0.0,
            ds, de, abs(ds), abs(de), ds * ds, de * de,
            span[0] / max(1.0, duration), span[1] / max(1.0, duration),
            temporal_iou(tuple(baseline), tuple(span)),
            (prefix_probability[last + 1] - prefix_probability[first]) / count,
        ]

    spans = [list(baseline)]
    vectors = [features(baseline, *bounds)]
    seen = {tuple(round(value, 6) for value in baseline)}
    for first in range(lo, hi + 1):
        for last in range(first, hi + 1):
            raw = [float(words[first]["start"]), float(words[last]["end"])]
            span = adjusted_span(raw, (0.2, 0.0), duration)
            key = tuple(span)
            if span[1] <= span[0] or key in seen:
                continue
            actual_first = first
            while actual_first <= last and words[actual_first]["end"] <= span[0]:
                actual_first += 1
            if actual_first > last:
                continue
            seen.add(key)
            spans.append(span)
            vectors.append(features(span, actual_first, last))
    matrix = np.asarray(vectors, dtype=np.float64)
    matrix -= matrix[0].copy()
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite span features.")
    return spans, matrix


def build_cases(records, transcript_dir, context_words):
    rows = {row["question_id"]: row for row in load_rows()}
    if len(records) != len(rows) or {row["question_id"] for row in records} != rows.keys():
        raise ValueError("A complete, unique set of supplied questions is required.")
    transcripts = {}
    cases = []
    for record in records:
        truth = rows[record["question_id"]]
        gold = gold_evidence(truth)
        if (
            record["transcript_id"] != truth["transcript_id"]
            or record["label"] != int(truth["label"])
            or record["gold"] != (list(gold) if gold is not None else None)
        ):
            raise ValueError(f"Reference data mismatch for {record['question_id']}.")
        tid = record["transcript_id"]
        if tid not in transcripts:
            transcripts[tid] = json.loads((transcript_dir / f"{tid}.json").read_text())
        transcript = transcripts[tid]
        spans, matrix = candidates(
            truth["question"], transcript["words"],
            record["span"] if record["answer"] else None,
            float(transcript["duration"]), context_words,
        )
        targets = np.asarray([
            temporal_iou(tuple(record["gold"]), tuple(span) if span is not None else None)
            if record["label"] == 1 else 0.0
            for span in spans
        ])
        delta = targets - targets[0]
        cases.append(Case(
            record, spans, matrix, targets,
            matrix.T @ matrix / len(spans), matrix.T @ delta / len(spans),
        ))
    return cases


def fit(cases, alpha):
    positives = [case for case in cases if case.row["label"] == 1]
    if not positives:
        raise ValueError("Training fold has no positives.")
    gram = sum((case.gram for case in positives), np.zeros((len(FEATURES), len(FEATURES))))
    cross = sum((case.cross for case in positives), np.zeros(len(FEATURES)))
    weights = np.linalg.solve(
        gram / len(positives) + alpha * np.eye(len(FEATURES)),
        cross / len(positives),
    )
    if not np.isfinite(weights).all():
        raise ValueError("Non-finite fitted span utility.")
    return weights


def select(case, weights, margin):
    predicted = case.features @ weights
    best = int(np.argmax(predicted))
    return best if predicted[best] > margin else 0


def choose_policy(cases, seed):
    tids = {case.row["transcript_id"] for case in cases}
    folds = grouped_folds(tids, min(3, len(tids)), seed)
    utility = {(alpha, margin): 0.0 for alpha in ALPHAS for margin in MARGINS}
    for held_out in folds:
        held = set(held_out)
        train = [case for case in cases if case.row["transcript_id"] not in held]
        validation = [case for case in cases if case.row["transcript_id"] in held]
        for alpha in ALPHAS:
            weights = fit(train, alpha)
            for margin in MARGINS:
                utility[(alpha, margin)] += sum(
                    case.targets[select(case, weights, margin)] - case.targets[0]
                    for case in validation if case.row["label"] == 1
                )
    return max(utility, key=lambda pair: (utility[pair], pair[0], pair[1]))


def evaluate(cases, seed):
    tids = {case.row["transcript_id"] for case in cases}
    folds = grouped_folds(tids, min(5, len(tids)), seed)
    records, reports = [], []
    for index, held_out in enumerate(folds):
        held = set(held_out)
        train = [case for case in cases if case.row["transcript_id"] not in held]
        test = [case for case in cases if case.row["transcript_id"] in held]
        alpha, margin = choose_policy(train, seed + index + 1)
        weights = fit(train, alpha)
        for case in test:
            selected = select(case, weights, margin)
            records.append({
                **case.row, "span": case.spans[selected],
                "original_span": case.row["span"], "changed": selected != 0,
                "selected_candidate": selected,
            })
        reports.append({
            "train_tids": sorted({case.row["transcript_id"] for case in train}),
            "held_out_tids": held_out, "alpha": alpha, "margin": margin,
        })
    return records, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--context-words", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 37])
    parser.add_argument("--output", default="results/span_ranker")
    args = parser.parse_args()
    if args.context_words < 0:
        parser.error("--context-words must be nonnegative.")
    record_path = Path(args.records)
    records = json.loads(record_path.read_text())
    requests = json.loads(Path(args.requests).read_text())
    if any("demonstration_tids" not in request for request in requests):
        parser.error("Demonstration provenance is required.")
    excluded = {tid for request in requests for tid in request["demonstration_tids"]}
    cases = build_cases(records, Path(args.transcripts), args.context_words)
    selected = [case for case in cases if case.row["transcript_id"] not in excluded]
    baseline = [case.row for case in selected]
    positives = [case for case in selected if case.row["label"] == 1]
    summary = {
        "baseline": score_records(baseline),
        "excluded_demonstration_tids": sorted(excluded),
        "features": FEATURES, "context_words": args.context_words,
        "alphas": ALPHAS, "margins": MARGINS,
        "candidate_oracle_miou": float(np.mean([np.max(case.targets) for case in positives])),
        "candidate_count": sum(len(case.spans) for case in selected),
        "input_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "seeds": {},
    }
    output = Path(args.output)
    for seed in args.seeds:
        predictions, folds = evaluate(selected, seed)
        result = {
            "candidate": score_records(predictions),
            "paired": paired_comparison(baseline, predictions, seed=seed),
            "changed_questions": sum(row["changed"] for row in predictions),
            "folds": folds,
        }
        summary["seeds"][str(seed)] = result
        write_json(output / f"oof_seed{seed}.json", predictions)
        print(json.dumps({"seed": seed, **{k: v for k, v in result.items() if k != "folds"}}), flush=True)
    summary["qualification"] = "candidate" if all(
        result["paired"]["conversation_bootstrap_95pct"][0] > 0
        for result in summary["seeds"].values()
    ) else "rejected"
    write_json(output / "summary.json", summary)
    print(json.dumps({
        "baseline": summary["baseline"], "candidate_oracle_miou": summary["candidate_oracle_miou"],
        "candidate_count": summary["candidate_count"], "qualification": summary["qualification"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
