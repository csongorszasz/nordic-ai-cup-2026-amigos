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
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asr  # noqa: E402  (imports no heavy deps at module level)
import windows as windows_module  # noqa: E402
from local_evaluator import Statistics  # noqa: E402
from utils import (  # noqa: E402
    Span,
    gold_evidence,
    group_questions_by_conversation,
    load_sample_audio,
    temporal_iou,
)

Answerer = Callable[[str, List[windows_module.Window]], Tuple[bool, Optional[Span]]]

# Padding applied to a window's boundaries when testing whether the annotated
# span is really about matches that stretch of audio.
PAD_START = 0.2
PAD_END = 0.4


def answer_all_true(question: str, windows: List[windows_module.Window]):
    """Stub: the shipped baseline behaviour (floor)."""
    return True, None


def answer_all_false(question: str, windows: List[windows_module.Window]):
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


def _neighbourhood_oracle(words, windows, gold, neighbours: int = 1) -> float:
    """Best sub-range inside the best clause +/- ``neighbours`` clauses.

    This is the span ceiling our localization step can actually reach, since it
    searches within a clause neighbourhood rather than the whole transcript.
    """
    best = 0.0
    for index in range(len(windows)):
        lo = max(0, index - neighbours)
        hi = min(len(windows) - 1, index + neighbours)
        indices = range(windows[lo].first_word, windows[hi].last_word + 1)
        best = max(best, _best_subrange(words, indices, gold))
    return best


def _window_diagnostics(words: List[Dict], windows, rows) -> Dict:
    spans = [gold_evidence(row) for row in rows]
    spans = [s for s in spans if s is not None]
    empty = {"gold": 0, "covered": 0, "overlap": 0, "best": [], "padded": [],
             "wordrange": [], "localize": []}
    if not spans:
        return empty

    best, padded, wordrange, localize = [], [], [], []
    covered = overlap = 0
    for gold in spans:
        clause = max((temporal_iou(gold, w.span()) for w in windows), default=0.0)
        pad = max(
            (temporal_iou(gold, (w.start - PAD_START, w.end + PAD_END)) for w in windows),
            default=0.0,
        )
        best.append(clause)
        padded.append(pad)
        wordrange.append(_word_range_oracle(words, gold))
        localize.append(_neighbourhood_oracle(words, windows, gold, neighbours=1))
        if clause > 0:
            overlap += 1
        if any(w.start <= gold[0] and w.end >= gold[1] for w in windows):
            covered += 1

    return {"gold": len(spans), "covered": covered, "overlap": overlap,
            "best": best, "padded": padded, "wordrange": wordrange,
            "localize": localize}


def run(
    answerer: Answerer,
    limit: Optional[int] = None,
    verbose: bool = False,
    diagnostics: bool = False,
    force_transcribe: bool = False,
) -> Statistics:
    statistics = Statistics()

    conversations = group_questions_by_conversation()
    if limit:
        conversations = conversations[:limit]

    window_counts: List[int] = []
    diag_accum = {"gold": 0, "covered": 0, "overlap": 0, "best": [], "padded": [],
                  "wordrange": [], "localize": []}

    for audio_filename, rows in conversations:
        audio_bytes = load_sample_audio(audio_filename)
        transcript = asr.transcribe_bytes(
            audio_bytes, audio_filename, cache=True, force=force_transcribe
        )
        windows = windows_module.windows_from_transcript(transcript)
        window_counts.append(len(windows))

        if diagnostics:
            diag = _window_diagnostics(transcript.get("words", []), windows, rows)
            for key in ("gold", "covered", "overlap"):
                diag_accum[key] += diag[key]
            for key in ("best", "padded", "wordrange", "localize"):
                diag_accum[key].extend(diag[key])

        for row in rows:
            label = int(row["label"])
            gold = gold_evidence(row)
            try:
                answer, span = answerer(row["question"], windows)
            except Exception:
                answer, span = True, None

            iou = statistics.record(
                row["question_type"], label, int(answer), gold, span
            )

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

    return statistics


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
        ("subrange in clause +/-1 (target)", localize),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only score the first N conversations.")
    parser.add_argument("--verbose", action="store_true",
                        help="One line per question.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Report window counts and gold-span coverage.")
    parser.add_argument("--answer", choices=["true", "false"], default="true",
                        help="Stub answerer to use (default: always true).")
    parser.add_argument("--force-transcribe", action="store_true",
                        help="Ignore the transcript cache (needs a GPU).")
    args = parser.parse_args()

    answerer = answer_all_true if args.answer == "true" else answer_all_false

    statistics = run(
        answerer,
        limit=args.limit,
        verbose=args.verbose,
        diagnostics=args.diagnostics,
        force_transcribe=args.force_transcribe,
    )
    print(statistics.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
