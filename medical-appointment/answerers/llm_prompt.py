"""Prompt builders for the LLM ceiling probe (L0 zero-shot, L1 few-shot, L2 two-pass).

All pure functions so the prompt shape is unit-testable without a model. The
input never includes ``question_type`` (unavailable at evaluation time) and never
includes retrieved candidates (L0-L2 are transcript-only).

``MEDAPP_LLM_PROMPT`` selects a prompt variant (``base``, ``v1``, ``v2``,
``v3``); every builder takes an explicit ``variant`` argument for probes so the
environment never has to change mid-run.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

from .align import text_between

# Variant selected at serving time; probes pass it explicitly.
VARIANT = os.environ.get("MEDAPP_LLM_PROMPT", "base").strip().lower()

SYSTEM_PROMPT = (
    "You are a clinical evidence extraction assistant. You are given a "
    "timestamped transcript of a doctor-patient consultation and a list of "
    "yes/no questions. For each question decide whether the transcript "
    "establishes the answer, and for a yes quote the exact transcript text that "
    "supports it.\n"
    "Rules:\n"
    "- Use only the transcript; never use outside knowledge.\n"
    "- answer is exactly \"yes\" or \"no\".\n"
    "- evidence_quote must be copied VERBATIM as a contiguous span from the "
    "transcript. If answer is \"no\", evidence_quote is null.\n"
    "- Do not paraphrase, translate, or add ellipses.\n"
    "- Output ONLY a JSON object, no prose."
)

# v1 (Attribute-First): locate the evidence before deciding the answer.
ATTRIBUTE_FIRST_NOTE = (
    "Work evidence-first: for each question, find the exact supporting span in "
    "the transcript first, and only then decide the answer."
)
# v2 (minimal evidence): counter the over-long quotes (CAGE boundary overrun).
MINIMAL_RULE = (
    "- evidence_quote must be the SHORTEST contiguous span that fully "
    "establishes the answer; do not include surrounding context that is not "
    "needed.\n"
)
# v3 (occurrence disambiguation): the question is written from the annotated
# occurrence, so its wording points at the occurrence a human would cite.
OCCURRENCE_RULE = (
    "- If the same fact is stated more than once, quote the occurrence whose "
    "wording most closely matches the question.\n"
)

_VARIANT_RULES = {"v2": MINIMAL_RULE, "v3": OCCURRENCE_RULE}

SCHEMA_HINT = (
    'Return JSON exactly like:\n'
    '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"..."},'
    '{"id":"q02","answer":"no","evidence_quote":null}]}'
)

SCHEMA_HINT_V1 = (
    'Return JSON exactly like:\n'
    '{"answers":[{"id":"q01","evidence_quote":"...","answer":"yes"},'
    '{"id":"q02","evidence_quote":null,"answer":"no"}]}'
)


def system_prompt(variant: Optional[str] = None) -> str:
    """System prompt for a variant (defaults to ``MEDAPP_LLM_PROMPT``)."""
    selected = (variant or VARIANT).strip().lower()
    if selected == "v1":
        return SYSTEM_PROMPT.replace(
            "Rules:\n", "Rules:\n" + ATTRIBUTE_FIRST_NOTE + "\n", 1
        )
    rule = _VARIANT_RULES.get(selected)
    if rule:
        return SYSTEM_PROMPT.replace("Rules:\n", "Rules:\n" + rule, 1)
    return SYSTEM_PROMPT


def schema_hint(variant: Optional[str] = None) -> str:
    """Schema hint for a variant; v1 puts the evidence before the answer."""
    return SCHEMA_HINT_V1 if (variant or VARIANT).strip().lower() == "v1" else SCHEMA_HINT


def qid_for(index: int) -> str:
    """Stable id for the question at ``index`` (0-based) in request order."""
    return f"q{index + 1:02d}"


def serialize_transcript(transcript: Dict, max_segments: Optional[int] = None) -> str:
    """Segment-level lines with stable ids and timestamps."""
    segments = transcript.get("segments", [])
    if max_segments is not None:
        segments = segments[:max_segments]
    return "\n".join(
        f"[s{seg['id']:02d} {seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}"
        for seg in segments
    )


def serialize_excerpt(
    transcript: Dict,
    start: Optional[float] = None,
    end: Optional[float] = None,
    pad: float = 6.0,
    max_segments: int = 8,
) -> str:
    """A short excerpt around a span, or the first few segments."""
    segments = transcript.get("segments", [])
    if start is not None and end is not None:
        chosen = [
            seg for seg in segments
            if seg["end"] >= start - pad and seg["start"] <= end + pad
        ]
    else:
        chosen = []
    if not chosen:
        chosen = segments[:max_segments]
    chosen = chosen[:max_segments]
    return "\n".join(
        f"[s{seg['id']:02d} {seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}"
        for seg in chosen
    )


def questions_block(questions: Sequence[str]) -> str:
    return "\n".join(
        f"{qid_for(index)}: {question}"
        for index, question in enumerate(questions)
    )


def _base_user(
    transcript: Dict, questions: Sequence[str], variant: Optional[str] = None
) -> str:
    return (
        "TRANSCRIPT\n"
        f"{serialize_transcript(transcript)}\n\n"
        "QUESTIONS\n"
        f"{questions_block(questions)}\n\n"
        f"{schema_hint(variant)}"
    )


def build_l0_messages(
    transcript: Dict, questions: Sequence[str], variant: Optional[str] = None
) -> List[Dict]:
    """Zero-shot, single call."""
    return [
        {"role": "system", "content": system_prompt(variant)},
        {"role": "user", "content": _base_user(transcript, questions, variant)},
    ]


def build_l1_messages(
    transcript: Dict,
    questions: Sequence[str],
    few_shot: Sequence[Tuple[str, str]] = (),
    variant: Optional[str] = None,
) -> List[Dict]:
    """Few-shot, single call. ``few_shot`` is a list of (user, assistant) turns."""
    messages: List[Dict] = [{"role": "system", "content": system_prompt(variant)}]
    for user, assistant in few_shot:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append(
        {"role": "user", "content": _base_user(transcript, questions, variant)}
    )
    return messages


def build_l2_decide_messages(
    transcript: Dict, questions: Sequence[str], few_shot: Sequence[Tuple[str, str]] = ()
) -> List[Dict]:
    """Pass 1 of L2: booleans only."""
    user = (
        "TRANSCRIPT\n"
        f"{serialize_transcript(transcript)}\n\n"
        "QUESTIONS\n"
        f"{questions_block(questions)}\n\n"
        'Return JSON exactly like: {"answers":[{"id":"q01","answer":"yes"},'
        '{"id":"q02","answer":"no"}]}'
    )
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example_user, example_assistant in few_shot:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": example_assistant})
    messages.append({"role": "user", "content": user})
    return messages


def build_l2_cite_messages(
    transcript: Dict, yes_questions: Sequence[Tuple[str, str]]
) -> List[Dict]:
    """Pass 2 of L2: verbatim quotes for the questions decided yes.

    ``yes_questions`` is ``[(qid, question), ...]``.
    """
    lines = "\n".join(f"{qid}: {question}" for qid, question in yes_questions)
    user = (
        "TRANSCRIPT\n"
        f"{serialize_transcript(transcript)}\n\n"
        "The following questions have already been answered YES. For each, quote "
        "the exact transcript text that supports it.\n"
        f"{lines}\n\n"
        'Return JSON exactly like: {"answers":[{"id":"q01",'
        '"evidence_quote":"..."}]}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


RAG_SYSTEM_PROMPT = (
    "You are a clinical evidence extraction assistant. For each yes/no question "
    "you are given a small set of candidate passages from the consultation "
    "transcript. Decide the answer, and for a yes choose the single candidate "
    "that best supports it and quote a contiguous span from within that "
    "candidate.\n"
    "Rules:\n"
    "- Use only the candidate passages; never use outside knowledge.\n"
    "- answer is exactly \"yes\" or \"no\".\n"
    "- On yes, candidate is a candidate id (e.g. \"c02\") and evidence_quote is "
    "copied VERBATIM from that candidate.\n"
    "- On no, candidate and evidence_quote are null.\n"
    "- Output ONLY a JSON object, no prose."
)

RAG_SCHEMA_HINT = (
    'Return JSON exactly like:\n'
    '{"answers":[{"id":"q01","answer":"yes","candidate":"c01",'
    '"evidence_quote":"..."},{"id":"q02","answer":"no","candidate":null,'
    '"evidence_quote":null}]}'
)


def candidate_ids(candidates_by_index) -> Dict[str, Tuple[int, int]]:
    """Globally unique candidate ids across all questions -> (q_index, c_index).

    Ids must not repeat per question, or the model cannot name which candidate
    it means and collapses to the first one.
    """
    mapping: Dict[str, Tuple[int, int]] = {}
    counter = 1
    for q_index, candidates in enumerate(candidates_by_index):
        for c_index in range(len(candidates)):
            mapping[f"c{counter:02d}"] = (q_index, c_index)
            counter += 1
    return mapping


def rag_candidates_block(questions: Sequence[str], candidates_by_index) -> str:
    lines: List[str] = []
    counter = 1
    for index, question in enumerate(questions):
        lines.append(f"{qid_for(index)}: {question}")
        for passage in candidates_by_index[index]:
            lines.append(
                f"  c{counter:02d} [{passage.start:.2f}-{passage.end:.2f}] {passage.text}"
            )
            counter += 1
    return "\n".join(lines)


def build_rag_messages(
    questions: Sequence[str], candidates_by_index, few_shot: Sequence[Tuple[str, str]] = ()
) -> List[Dict]:
    """Grounded RAG reader: answer + choose a candidate + quote from it."""
    last = qid_for(len(questions) - 1) if questions else "q00"
    user = (
        f"Answer EVERY question below ({qid_for(0)}..{last}); output exactly one "
        "JSON entry per question, in order, even when the answer is no. Each "
        "candidate has a unique id (c01, c02, ...) and the candidate you cite "
        "must be listed under that same question.\n\n"
        "CANDIDATES\n"
        f"{rag_candidates_block(questions, candidates_by_index)}\n\n"
        f"{RAG_SCHEMA_HINT}"
    )
    messages: List[Dict] = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    for example_user, example_assistant in few_shot:
        messages.append({"role": "user", "content": example_user})
        messages.append({"role": "assistant", "content": example_assistant})
    messages.append({"role": "user", "content": user})
    return messages


def render_rag_example(
    question: str,
    candidates,
    answer: bool,
    candidate_id,
    quote,
) -> Tuple[str, str]:
    """One RAG few-shot turn pair: candidates + question -> JSON with citation."""
    import json

    user = (
        "CANDIDATES\n"
        f"{rag_candidates_block([question], [candidates])}\n\n"
        f"{RAG_SCHEMA_HINT}"
    )
    assistant = json.dumps(
        {
            "answers": [
                {
                    "id": "q01",
                    "answer": "yes" if answer else "no",
                    "candidate": candidate_id if answer else None,
                    "evidence_quote": quote if answer else None,
                }
            ]
        }
    )
    return user, assistant


def render_example(
    transcript: Dict,
    question: str,
    answer: bool,
    quote: Optional[str],
    evidence_span: Optional[Tuple[float, float]] = None,
) -> Tuple[str, str]:
    """One few-shot turn pair from an example conversation."""
    start, end = evidence_span if evidence_span else (None, None)
    excerpt = serialize_excerpt(transcript, start, end)
    user = (
        "TRANSCRIPT\n"
        f"{excerpt}\n\n"
        "QUESTIONS\n"
        f"q01: {question}\n\n"
        f"{SCHEMA_HINT}"
    )
    import json

    assistant = json.dumps(
        {
            "answers": [
                {
                    "id": "q01",
                    "answer": "yes" if answer else "no",
                    "evidence_quote": quote if answer else None,
                }
            ]
        }
    )
    return user, assistant


def build_few_shot(
    rows_by_tid: Dict[str, List[Dict]],
    transcripts: Dict[str, Dict],
    evidence: Dict[str, Dict],
    exclude_tid: str,
    counts: Tuple[int, int, int] = (1, 1, 1),
) -> List[Tuple[str, str]]:
    """Balanced LOCO-safe examples: positive, hard-negative refute, off-topic.

    Draws from conversations other than ``exclude_tid`` so the target's answer
    can never leak into its own prompt.
    """
    want_positive, want_refute, want_offtopic = counts
    examples: List[Tuple[str, str]] = []
    used = {exclude_tid}
    tids = sorted(rows_by_tid)

    def find(question_type: str, require_refute: bool = False) -> Optional[Dict]:
        for tid in tids:
            if tid in used or tid not in transcripts:
                continue
            for row in rows_by_tid[tid]:
                if row["question_type"] != question_type:
                    continue
                item = evidence.get(row["question_id"], {})
                if require_refute and item.get("bucket") != "refute":
                    continue
                return row
        return None

    def add(row: Optional[Dict]) -> None:
        if row is None:
            return
        tid = row["transcript_id"]
        used.add(tid)
        words = transcripts[tid]["words"]
        item = evidence.get(row["question_id"], {})
        if row["question_type"] == "positive":
            span = (float(row["evidence_start"]), float(row["evidence_end"]))
            quote = text_between(words, span[0], span[1])
            examples.append(render_example(
                transcripts[tid], row["question"], True, quote, span
            ))
        elif row["question_type"] == "hard_negative":
            span = None
            if item.get("start") and item.get("end"):
                span = (float(item["start"]), float(item["end"]))
            examples.append(render_example(
                transcripts[tid], row["question"], False, None, span
            ))
        else:
            examples.append(render_example(
                transcripts[tid], row["question"], False, None, None
            ))

    for _ in range(want_positive):
        add(find("positive"))
    for _ in range(want_refute):
        add(find("hard_negative", require_refute=True))
    for _ in range(want_offtopic):
        add(find("off_topic"))
    return examples
