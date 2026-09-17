"""Score the pipeline in-process, without running the HTTP server.

This replays the supplied conversations through the same functions ``example.py``
will use and scores them with the official ``local_evaluator.Statistics``, so the
numbers mean the same thing as a real attempt. Transcripts are cached, so after
the first run iteration is verifier-only and fast — and diagnostics can be run
locally off the cache with no GPU.

    python dev_eval.py                     # stub answerer over all 39
    python dev_eval.py --diagnostics       # + window / gold-span coverage
    python dev_eval.py --limit 3 --verbose
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import answer as answer_module  # noqa: E402  (imports no heavy deps at module level)
import asr  # noqa: E402
import windows as windows_module  # noqa: E402
from local_evaluator import Statistics  # noqa: E402
from utils import (  # noqa: E402
    Span,
    gold_evidence,
    group_questions_by_conversation,
    load_sample_audio,
    temporal_iou,
)

Answerer = Callable[[str, List[Dict], List[windows_module.Window]],
                    Tuple[bool, Optional[Span]]]

# Padding applied to a window's boundaries when testing whether the annotated
# span is really about matches that stretch of audio.
PAD_START = 0.2
PAD_END = 0.4


def answer_all_true(question: str, words: List[Dict], windows):
    """Stub: the shipped baseline behaviour (floor)."""
    return True, None


def answer_all_false(question: str, words: List[Dict], windows):
    """Stub: the other floor."""
    return False, None


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


def _best_subrange(words: List[Dict], indices, gold: Span, pad: float = 2.0) -> float:
    """Best tIoU over contiguous word ranges drawn from ``indices``."""
    lo, hi = gold[0] - pad, gold[1] + pad
    candidates = [
        k for k in indices if words[k]["end"] >= lo and words[k]["start"] <= hi
    ]
    best = 0.0
    for a in range(len(candidates)):
        i = candidates[a]
        for b in range(a, len(candidates)):
            j = candidates[b]
            span = (words[i]["start"], words[j]["end"])
            if span[1] < gold[0] or span[0] > gold[1]:
                continue
            best = max(best, temporal_iou(gold, span))
    return best


def _word_range_oracle(words: List[Dict], gold: Span, pad: float = 2.0) -> float:
    """Best tIoU over any contiguous word range — the ceiling windowing can reach."""
    return _best_subrange(words, range(len(words)), gold, pad=pad)


def _pruned_candidate_oracle(words, windows, question, gold, neighbours: int = 1) -> float:
    """Best tIoU over the actual candidate set the localizer searches.

    Same enumeration and pruning ``answer.py`` uses (clause neighbourhood plus
    a content-token filter), so this is the span ceiling the selector can reach.
    """
    content = answer_module._content_tokens(question)
    best = 0.0
    for index in range(len(windows)):
        lo = max(0, index - neighbours)
        hi = min(len(windows) - 1, index + neighbours)
        first = windows[lo].first_word
        last = windows[hi].last_word
        for i, j in answer_module._enumerate_candidates(words, first, last, content):
            span = (words[i]["start"], words[j]["end"])
            if span[1] < gold[0] or span[0] > gold[1]:
                continue
            best = max(best, temporal_iou(gold, span))
    return best


def _window_diagnostics(words: List[Dict], windows, rows) -> Dict:
    empty = {"gold": 0, "covered": 0, "overlap": 0, "best": [], "padded": [],
             "wordrange": [], "localize": []}
    best, padded, wordrange, localize = [], [], [], []
    covered = overlap = 0
    gold_count = 0
    for row in rows:
        gold = gold_evidence(row)
        if gold is None:
            continue
        gold_count += 1
        clause = max((temporal_iou(gold, w.span()) for w in windows), default=0.0)
        pad = max(
            (temporal_iou(gold, (w.start - PAD_START, w.end + PAD_END)) for w in windows),
            default=0.0,
        )
        best.append(clause)
        padded.append(pad)
        wordrange.append(_word_range_oracle(words, gold))
        localize.append(
            _pruned_candidate_oracle(words, windows, row["question"], gold)
        )
        if clause > 0:
            overlap += 1
        if any(w.start <= gold[0] and w.end >= gold[1] for w in windows):
            covered += 1

    if not gold_count:
        return empty
    return {"gold": gold_count, "covered": covered, "overlap": overlap,
            "best": best, "padded": padded, "wordrange": wordrange,
            "localize": localize}


def _span_text(words: List[Dict], span: Span) -> str:
    lo, hi = span
    tokens = [w["word"] for w in words if w["end"] > lo and w["start"] < hi]
    return " ".join("".join(tokens).split())


def _print_example(row, words, windows, info, gold, span, iou) -> None:
    best = info.get("best_clause")
    clause = windows[best].text if best is not None else ""
    print("\n  ---- low-IoU positive ----")
    print(f"    Q:        {row['question']}")
    print(f"    gold:     [{gold[0]:.2f}-{gold[1]:.2f}] {_span_text(words, gold)}")
    print(
        f"    chosen:   [{span[0]:.2f}-{span[1]:.2f}] {_span_text(words, span)}"
        f"   tIoU {iou:.3f}"
    )
    print(f"    clause:   {clause}")
    print(
        f"    candidates={info.get('n_candidates')} chosen={info.get('chosen')} "
        f"clause_score={info.get('clause_score')}"
    )


def run(
    answerer: Answerer,
    limit: Optional[int] = None,
    verbose: bool = False,
    diagnostics: bool = False,
    force_transcribe: bool = False,
    debug_examples: int = 0,
) -> Statistics:
    statistics = Statistics()

    conversations = group_questions_by_conversation()
    if limit:
        conversations = conversations[:limit]

    window_counts: List[int] = []
    nli_mode = answerer is answer_module.answer_question
    if nli_mode:
        from verifier import nli as _nli

        _nli.warm_up()  # keep model load out of the per-request timings
    debug_printed = 0
    timing = {"asr": [], "windows": [], "answer": [], "clause": [], "localize": [],
              "clause_pairs": 0, "candidate_pairs": 0, "trim_pairs": 0,
              "cache_hits": 0, "conversations": 0}
    diag_accum = {"gold": 0, "covered": 0, "overlap": 0, "best": [], "padded": [],
                  "wordrange": [], "localize": [], "chosen": [], "neigh": [],
                  "decided_by": {}}

    for audio_filename, rows in conversations:
        audio_bytes = load_sample_audio(audio_filename)
        cache_hit = (not force_transcribe) and asr.cache_path(audio_filename).exists()

        started = time.perf_counter()
        transcript = asr.transcribe_bytes(
            audio_bytes, audio_filename, cache=True, force=force_transcribe
        )
        timing["asr"].append(time.perf_counter() - started)
        timing["cache_hits"] += 1 if cache_hit else 0
        timing["conversations"] += 1

        started = time.perf_counter()
        windows = windows_module.windows_from_transcript(transcript)
        timing["windows"].append(time.perf_counter() - started)
        words = transcript.get("words", [])
        window_counts.append(len(windows))

        if diagnostics:
            diag = _window_diagnostics(words, windows, rows)
            for key in ("gold", "covered", "overlap"):
                diag_accum[key] += diag[key]
            for key in ("best", "padded", "wordrange", "localize"):
                diag_accum[key].extend(diag[key])

        for row in rows:
            label = int(row["label"])
            gold = gold_evidence(row)
            info = None
            started = time.perf_counter()
            try:
                if nli_mode:
                    answer, span, info = answerer(
                        row["question"], words, windows, return_info=True
                    )
                else:
                    answer, span = answerer(row["question"], words, windows)
            except Exception:
                answer, span = True, None
            timing["answer"].append(time.perf_counter() - started)

            if info:
                timing["clause"].append(info.get("t_clause", 0.0))
                timing["localize"].append(info.get("t_localize", 0.0))
                timing["clause_pairs"] += info.get("n_clause_pairs", 0)
                timing["candidate_pairs"] += info.get("n_candidate_pairs", 0)
                timing["trim_pairs"] += info.get("n_trim_pairs", 0)

            iou = statistics.record(
                row["question_type"], label, int(answer), gold, span
            )

            if diagnostics and nli_mode and gold is not None and answer and span:
                diag_accum["chosen"].append(temporal_iou(gold, span))
                if info and info.get("neighbourhood"):
                    first, last = info["neighbourhood"]
                    diag_accum["neigh"].append(
                        _best_subrange(words, range(first, last + 1), gold)
                    )
            if diagnostics and nli_mode and info:
                decided = info.get("decided_by") or "unknown"
                diag_accum["decided_by"][decided] = (
                    diag_accum["decided_by"].get(decided, 0) + 1
                )

            if (
                debug_examples
                and nli_mode
                and gold is not None
                and answer
                and span is not None
                and iou < 0.4
                and debug_printed < debug_examples
            ):
                debug_printed += 1
                _print_example(row, words, windows, info, gold, span, iou)

            if verbose:
                mark = "ok  " if int(answer) == label else "WRONG"
                evidence = f" tIoU {iou:.3f}" if gold is not None else ""
                print(
                    f"  {mark} {row['question_id']:<24} {row['question_type']:<14}"
                    f" said {'yes' if answer else 'no':<3} wanted {row['answer']:<3}"
                    f"{evidence}"
                )

        statistics.record_request(len(rows), None, failed=False)

    if diagnostics:
        _print_diagnostics(window_counts, diag_accum)
    if nli_mode:
        _print_timing(timing)

    statistics.diagnostics = diag_accum
    statistics.window_counts = window_counts
    statistics.timing = timing
    return statistics


def _worst(values: List[float]) -> float:
    return max(values) if values else 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _print_timing(timing: Dict) -> None:
    conversations = timing["conversations"] or 1
    asr = timing["asr"]
    windows = timing["windows"]
    answer = timing["answer"]
    clause = timing["clause"]
    localize = timing["localize"]

    print("\nTiming (NLI model pre-loaded; transcripts cached)")
    print(
        f"  ASR                mean {_mean(asr):6.2f}s  worst {_worst(asr):6.2f}s  "
        f"(cache hits {timing['cache_hits']}/{timing['conversations']})"
    )
    print(f"  windows            mean {_mean(windows):6.3f}s")
    print(
        f"  answer per conv    mean {sum(answer) / conversations:6.2f}s"
        f"   per question {_mean(answer):6.2f}s   worst question {_worst(answer):6.2f}s"
    )
    print(f"    decision NLI     mean {sum(clause) / conversations:6.2f}s (per conversation)")
    print(f"    localization NLI mean {sum(localize) / conversations:6.2f}s (per conversation)")
    print(
        f"  NLI pairs/conv     clause {timing['clause_pairs'] / conversations:.0f}  "
        f"candidates {timing['candidate_pairs'] / conversations:.0f}  "
        f"trim {timing['trim_pairs'] / conversations:.0f}  "
        f"(total {(timing['clause_pairs'] + timing['candidate_pairs'] + timing['trim_pairs']) / conversations:.0f})"
    )


def _print_diagnostics(window_counts: List[int], diag: Dict) -> None:
    print("\nWindow / gold-span diagnostics")
    if window_counts:
        print(
            f"  windows per conversation  {sum(window_counts) / len(window_counts):.1f} "
            f"mean, {min(window_counts)} min, {max(window_counts)} max, "
            f"{sum(window_counts)} total"
        )
    gold = diag["gold"]
    if not gold:
        return
    best, padded, wordrange = diag["best"], diag["padded"], diag["wordrange"]
    localize = diag["localize"]
    print(f"  gold spans                                 {gold}")
    print(
        f"  fully inside some window                   "
        f"{diag['covered']}/{gold} ({diag['covered'] / gold:.1%})"
    )
    print(
        f"  overlapping some window                    "
        f"{diag['overlap']}/{gold} ({diag['overlap'] / gold:.1%})"
    )
    for name, values in (
        ("clause-window oracle", best),
        ("pruned-candidate oracle (target)", localize),
        ("word-range oracle (upper bound)", wordrange),
        (f"padded-window oracle (+{PAD_START}/+{PAD_END}s)", padded),
    ):
        mean = sum(values) / len(values)
        ge50 = sum(1 for v in values if v >= 0.5)
        ge80 = sum(1 for v in values if v >= 0.8)
        perfect = sum(1 for v in values if v >= 0.999)
        print(
            f"  {name:<41} mean {mean:.3f}  "
            f">=0.5 {ge50}  >=0.8 {ge80}  ==1 {perfect}"
        )

    chosen = diag.get("chosen", [])
    neigh = diag.get("neigh", [])
    if chosen or neigh:
        print("  Localization (positives answered yes)")
        print(
            f"    chosen span mean tIoU              {_mean(chosen):.3f}  "
            f"(n={len(chosen)})"
        )
        print(
            f"    searched-neighbourhood oracle      {_mean(neigh):.3f}  "
            f"(n={len(neigh)})"
        )
        print(f"    decisions by path                  {diag.get('decided_by')}")


def _summary_to_json(statistics: Statistics, tag: str, elapsed: float) -> Dict:
    by_type = {
        name: {"correct": stats[0], "total": stats[1]}
        for name, stats in statistics.by_type.items()
    }
    diag = getattr(statistics, "diagnostics", {})
    timing = getattr(statistics, "timing", {})
    conversations = timing.get("conversations") or 0
    per_conv = (lambda values: sum(values) / conversations) if conversations else (lambda values: 0.0)
    return {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "accuracy": round(statistics.accuracy, 4),
        "mean_tiou": round(statistics.mean_tiou, 4),
        "score": round(statistics.final_score, 4),
        "questions": statistics.total,
        "missing_spans": statistics.missing_spans,
        "by_type": by_type,
        "window_oracle_clause": round(_mean(diag.get("best", [])), 4),
        "window_oracle_pruned": round(_mean(diag.get("localize", [])), 4),
        "window_oracle_upper": round(_mean(diag.get("wordrange", [])), 4),
        "timing": {
            "conversations": conversations,
            "cache_hits": timing.get("cache_hits", 0),
            "asr_mean_s": round(per_conv(timing.get("asr", [])), 2),
            "answer_mean_s": round(per_conv(timing.get("answer", [])), 2),
            "answer_worst_s": round(_worst(timing.get("answer", [])), 2),
            "decision_mean_s": round(per_conv(timing.get("clause", [])), 2),
            "localization_mean_s": round(per_conv(timing.get("localize", [])), 2),
            "clause_pairs_per_conv": round(
                timing.get("clause_pairs", 0) / conversations if conversations else 0, 1
            ),
            "candidate_pairs_per_conv": round(
                timing.get("candidate_pairs", 0) / conversations if conversations else 0, 1
            ),
            "trim_pairs_per_conv": round(
                timing.get("trim_pairs", 0) / conversations if conversations else 0, 1
            ),
        },
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N conversations.")
    parser.add_argument("--verbose", action="store_true",
                        help="One line per question.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Report window counts and gold-span coverage.")
    parser.add_argument("--answer", choices=["true", "false", "nli"], default="true",
                        help="Answerer: always true/false, or the real NLI pipeline.")
    parser.add_argument("--force-transcribe", action="store_true",
                        help="Ignore the transcript cache (needs a GPU).")
    parser.add_argument("--tag", default=None,
                        help="Name for this trial (used in the JSON output).")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Write a machine-readable summary to this path.")
    parser.add_argument("--debug", type=int, default=0,
                        help="Print N low-IoU positive examples.")
    args = parser.parse_args()

    if args.answer == "true":
        answerer = answer_all_true
    elif args.answer == "false":
        answerer = answer_all_false
    else:
        answerer = answer_module.answer_question

    started = time.time()
    statistics = run(
        answerer,
        limit=args.limit,
        verbose=args.verbose,
        diagnostics=args.diagnostics,
        force_transcribe=args.force_transcribe,
        debug_examples=args.debug,
    )
    elapsed = time.time() - started
    print(statistics.report())

    if args.json_path:
        tag = args.tag or Path(args.json_path).stem
        summary = _summary_to_json(statistics, tag, elapsed)
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(summary, indent=2))
        print(f"\nsummary -> {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
