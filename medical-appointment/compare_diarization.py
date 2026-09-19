"""Paired comparison for the speaker-tagged vs untagged LLM probe.

Development-only. Reads the two per-question record files written by
``llm_probe.py`` and reports the composite-score delta with a
conversation-resampled 95% interval, both raw and after the fixed +0.2 s
boundary correction, on all questions and on the demonstration-disjoint cohort.

    python compare_diarization.py \
        --base results/llm_dia_base_L1_questions.json \
        --speaker results/llm_dia_speaker_L1_questions.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from answerers.boundaries import adjusted_span
from answerers import llm_prompt
from answerers.llm_prompt import VARIANT, few_shot_counts
from answerers.modernbert_data import load_evidence, load_rows, load_transcript
from benchmark import paired_comparison

PROJECT_ROOT = Path(__file__).resolve().parent


def demonstration_tids(tids, modes=("first", "similar")):
    """Every conversation used as a few-shot source under the given selection modes."""
    rows_by_tid = defaultdict(list)
    for row in load_rows():
        rows_by_tid[row["transcript_id"]].append(row)
    transcripts = {tid: load_transcript(tid) for tid in rows_by_tid}
    evidence = load_evidence()
    original = llm_prompt.FEWSHOT_SELECT
    demo = set()
    try:
        for mode in modes:
            llm_prompt.FEWSHOT_SELECT = mode
            for tid in tids:
                for row in llm_prompt.select_few_shot_rows(
                    rows_by_tid, transcripts, evidence, tid, few_shot_counts(VARIANT)
                ):
                    demo.add(row["transcript_id"])
    finally:
        llm_prompt.FEWSHOT_SELECT = original
    return demo


def with_fixed_offset(records):
    out = []
    for record in records:
        clone = dict(record)
        if clone.get("span"):
            clone["span"] = list(adjusted_span(clone["span"], (0.2, 0.0), None))
        out.append(clone)
    return out


def subset(records, keep):
    return [record for record in records if record["transcript_id"] in keep]


def report(label, baseline, candidate):
    result = paired_comparison(baseline, candidate)
    lo, hi = result["conversation_bootstrap_95pct"]
    print(
        f"  {label:<28} delta {result['score_delta']:+.5f}  "
        f"95% CI [{lo:+.5f}, {hi:+.5f}]  conv={result['conversations']}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--speaker", type=Path, required=True)
    parser.add_argument("--demo-modes", default="first,similar",
                        help="Few-shot selection modes whose sources form the demo set.")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    speaker = json.loads(args.speaker.read_text())
    if {r["question_id"] for r in base} != {r["question_id"] for r in speaker}:
        raise ValueError("The two runs must cover the same questions.")

    tids = sorted({r["transcript_id"] for r in base})
    demo = demonstration_tids(tids, tuple(m.strip() for m in args.demo_modes.split(",") if m.strip()))
    non_demo = set(tids) - demo
    print(f"conversations={len(tids)} demonstration={len(demo)} non-demo={len(non_demo)}")

    for name, transform in (("raw", lambda rows: rows), ("fixed+0.2", with_fixed_offset)):
        baseline = transform(base)
        candidate = transform(speaker)
        print(f"{name}:")
        report("all questions", baseline, candidate)
        report("demonstration-disjoint", subset(baseline, non_demo), subset(candidate, non_demo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
