"""Bounded localization-only LLM probe with fixed decisions and baseline retention."""

import argparse
import json
import logging
import math
import re
import time
from pathlib import Path

from answerers.align import align_quote_matches, text_between
from answerers.boundaries import adjusted_span
from answerers.base import normalize_answer
from answerers.llm_client import HFClient
from answerers.llm_parse import parse_answers
from answerers.llm_prompt import qid_for
from answerers.passages import overlap_word_range
from benchmark import paired_comparison, write_json
from llm_probe import score_records
from utils import temporal_iou
from windows import join_words

logger = logging.getLogger(__name__)
SYSTEM = (
    "Refine evidence citations, not answers. Every listed question has already "
    "been answered yes. Each has a current quote and a local transcript context. "
    "Follow the evidence-selection convention demonstrated by the examples. "
    "Keep a precise current citation; otherwise copy the relevant contiguous "
    "transcript phrase, retaining its necessary qualifiers. A citation may be "
    "a fragment: surrounding context supplies meaning. Never paraphrase or "
    "invent words. A changed quote must identify a unique occurrence within "
    "its supplied context. Return JSON only."
)
SCHEMA = (
    'Return {"answers":[{"id":"q01","keep":true},'
    '{"id":"q02","keep":false,"evidence_quote":"exact local transcript text"}]}. '
    "Return one entry for every listed id."
)
BLIND_SYSTEM = (
    "Select evidence citations, not answers. Every listed question has already "
    "been answered yes. For each question, copy a contiguous phrase from its "
    "provided local transcript context, following the reference convention in "
    "the examples. Preserve necessary qualifiers. The quote may be a fragment "
    "because surrounding context supplies meaning. Never paraphrase or invent "
    "words. Identify a unique occurrence within that context. Return JSON only."
)
BLIND_SCHEMA = (
    'Return {"answers":[{"id":"q01","evidence_quote":"exact local transcript text"}]}. '
    "Return one entry for every listed id."
)
POINTER_SYSTEM = (
    "Select evidence boundaries, not answers. Every listed question has already "
    "been answered yes. For each local word-indexed transcript context, identify "
    "the first and last word of the supporting evidence, following the examples. "
    "Preserve necessary qualifiers. Evidence may be a fragment because surrounding "
    "context supplies meaning. Use only word IDs from that question's context. "
    "Return JSON only."
)
POINTER_SCHEMA = (
    'Return {"answers":[{"id":"q01","first_word":"w03","last_word":"w08"}]}. '
    "Both word IDs are inclusive. Return one entry for every listed question."
)


def make_case(row, transcript, qid, context=24):
    words = transcript["words"]
    bounds = overlap_word_range(words, *row["span"])
    if bounds is None:
        raise ValueError(f"No words overlap the citation for {row['question_id']}.")
    first, last = max(0, bounds[0] - context), min(len(words) - 1, bounds[1] + context)
    return {
        "qid": qid, "row": row, "first": first, "last": last,
        "context": join_words(words, first, last),
        "tokens": [word["word"].strip() for word in words[first:last + 1]],
        "start": words[first]["start"], "end": words[last]["end"],
    }


def render(cases, blind=False, word_indices=False):
    blocks = []
    for case in cases:
        lines = [f"{case['qid']}: {case['row']['question']}"]
        if not blind and not word_indices:
            lines.append(f"CURRENT: {case['row']['quote']}")
        context = (
            " ".join(f"[w{index:02d}] {word}" for index, word in enumerate(case["tokens"]))
            if word_indices else case["context"]
        )
        lines.append(f"CONTEXT [{case['start']:.2f}-{case['end']:.2f}]: {context}")
        blocks.append("\n".join(lines))
    schema = POINTER_SCHEMA if word_indices else BLIND_SCHEMA if blind else SCHEMA
    return "\n\n".join(blocks) + "\n\n" + schema


def build_examples(rows, transcripts, pool, exclude_tid, blind=False, word_indices=False):
    examples, sources = [], []
    for want_keep in (True, False):
        for row in rows:
            tid = row["transcript_id"]
            if tid not in pool or tid == exclude_tid or tid in sources:
                continue
            if row["label"] != 1 or not row["answer"] or not row.get("quote"):
                continue
            quality = temporal_iou(tuple(row["gold"]), tuple(row["span"]))
            if not (quality >= 0.8 if want_keep else quality < 0.5):
                continue
            case = make_case(row, transcripts[tid], "q01")
            gold_range = overlap_word_range(transcripts[tid]["words"], *row["gold"])
            if gold_range is None or not case["first"] <= gold_range[0] <= gold_range[1] <= case["last"]:
                continue
            entry = {"id": "q01", "keep": want_keep}
            if word_indices:
                entry = {
                    "id": "q01",
                    "first_word": f"w{gold_range[0] - case['first']:02d}",
                    "last_word": f"w{gold_range[1] - case['first']:02d}",
                }
            elif blind:
                entry.pop("keep")
            if not word_indices and (blind or not want_keep):
                entry["evidence_quote"] = text_between(transcripts[tid]["words"], *row["gold"])
            examples.extend([
                {"role": "user", "content": render([case], blind, word_indices)},
                {"role": "assistant", "content": json.dumps({"answers": [entry]})},
            ])
            sources.append(tid)
            break
    return examples, sources


def decode_refinement(entry, case, transcript, word_indices=False):
    """Only unique local quotes can replace the baseline; decisions never change."""
    original = case["row"]["span"]
    baseline = adjusted_span(original, (0.2, 0.0), transcript.get("duration"))
    if entry is None:
        return baseline, "missing"
    if word_indices:
        indices = []
        for key in ("first_word", "last_word"):
            value = entry.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"w\d+", value):
                return baseline, "invalid_indices"
            indices.append(int(value[1:]))
        first, last = indices
        if not 0 <= first <= last <= case["last"] - case["first"]:
            return baseline, "invalid_indices"
        words = transcript["words"]
        proposal = adjusted_span(
            [words[case["first"] + first]["start"], words[case["first"] + last]["end"]],
            (0.2, 0.0), transcript.get("duration"),
        )
        valid, checked = normalize_answer((True, proposal), duration=transcript.get("duration"))
        return (list(checked), "refined") if valid else (baseline, "invalid_bounds")
    quote = entry.get("quote")
    if entry.get("keep") is True or quote == case["row"].get("quote"):
        return baseline, "kept"
    matches = align_quote_matches(
        transcript["words"], quote or "", first_word=case["first"], last_word=case["last"]
    )
    if len(matches) != 1:
        return baseline, "ambiguous" if matches else "unaligned"
    span = adjusted_span(matches[0][:2], (0.2, 0.0), transcript.get("duration"))
    return span, "refined"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/local_refinement"))
    parser.add_argument("--call-budget", type=float, default=20.0)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--word-indices", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.call_budget) or not 0 < args.call_budget <= 20:
        parser.error("--call-budget must be positive and no greater than 20 seconds.")
    logging.basicConfig(level=logging.INFO)
    rows = json.loads((args.baseline / "base_legacy_questions.json").read_text())
    requests = json.loads((args.baseline / "base_legacy_conversations.json").read_text())
    baseline_summary = json.loads((args.baseline / "base_legacy_summary.json").read_text())
    if (
        len(rows) != 390 or len({row["question_id"] for row in rows}) != 390
        or not baseline_summary.get("complete")
        or baseline_summary["model"] != args.model
        or baseline_summary["revision"] != args.revision
    ):
        parser.error("A complete baseline from the same frozen model is required.")
    pool = {tid for request in requests for tid in request["demonstration_tids"]}
    transcripts = {
        request["transcript_id"]: json.loads(
            (args.baseline / "transcripts" / f"{request['transcript_id']}.json").read_text()
        )
        for request in requests
    }
    client = HFClient(
        model_name=args.model, revision=args.revision, legacy_special_tokens=True,
        max_new_tokens=768,
    )
    client.warm_up()
    records, logs = [], []
    for request in requests:
        tid = request["transcript_id"]
        selected = [row for row in rows if row["transcript_id"] == tid]
        positives = [row for row in selected if row["answer"] and row["span"]]
        cases = [make_case(row, transcripts[tid], qid_for(index)) for index, row in enumerate(positives)]
        examples, sources = build_examples(rows, transcripts, pool, tid, args.blind, args.word_indices)
        system = POINTER_SYSTEM if args.word_indices else BLIND_SYSTEM if args.blind else SYSTEM
        messages = [{"role": "system", "content": system}, *examples,
                    {"role": "user", "content": render(cases, args.blind, args.word_indices)}]
        started = time.monotonic()
        generation_failed = False
        try:
            raw = client.generate(messages, deadline=started + args.call_budget) if cases else ""
        except Exception:
            logger.exception("Local refinement failed for %s; retaining the baseline.", tid)
            raw = ""
            generation_failed = True
        elapsed = time.monotonic() - started
        parsed = parse_answers(raw, [case["qid"] for case in cases])
        updates = {}
        for case in cases:
            span, reason = decode_refinement(
                parsed.get(case["qid"]), case, transcripts[tid], args.word_indices
            )
            updates[case["row"]["question_id"]] = (span, reason)
        for row in selected:
            span, reason = updates.get(row["question_id"], (None, "base_no"))
            records.append({**row, "span": span, "refinement": reason, "original_span": row["span"]})
        logs.append({
            "transcript_id": tid, "sources": sources, "raw_output": raw,
            "refinement_s": elapsed, "base_uncached_s": request["latency_s"],
            "estimated_combined_s": request["latency_s"] + elapsed,
            "generation_failed": generation_failed,
        })
        write_json(args.output / "questions.json", records)
        write_json(args.output / "conversations.json", logs)
        print(f"{tid}: {elapsed:.2f}s, {len(cases)} citations", flush=True)
    baseline = [
        {**row, "span": adjusted_span(row["span"], (0.2, 0.0), row.get("duration")) if row["answer"] else None}
        for row in rows
    ]
    safe_base = [row for row in baseline if row["transcript_id"] not in pool]
    safe_candidate = [row for row in records if row["transcript_id"] not in pool]
    reasons = {}
    for row in records:
        reasons[row["refinement"]] = reasons.get(row["refinement"], 0) + 1
    summary = {
        "complete": len(records) == len(rows) == 390,
        "baseline": score_records(baseline), "candidate": score_records(records),
        "non_demo_baseline": score_records(safe_base),
        "non_demo_candidate": score_records(safe_candidate),
        "paired": paired_comparison(safe_base, safe_candidate),
        "reasons": reasons, "excluded_demo_tids": sorted(pool),
        "max_refinement_s": max(item["refinement_s"] for item in logs),
        "max_estimated_combined_s": max(item["estimated_combined_s"] for item in logs),
        "generation_failures": sum(item["generation_failed"] for item in logs),
        "visible_anchor": not (args.blind or args.word_indices),
        "output_representation": "word_indices" if args.word_indices else "quote",
        "latency_note": "Sum of separate measurements, not an HTTP acceptance result.",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
