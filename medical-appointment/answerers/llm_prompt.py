"""Prompt builders for the LLM ceiling probe (L0 zero-shot, L1 few-shot, L2 two-pass).

All pure functions so the prompt shape is unit-testable without a model. The
input never includes ``question_type`` (unavailable at evaluation time) and never
includes retrieved candidates (L0-L2 are transcript-only).
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .align import text_between

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

SCHEMA_HINT = (
    'Return JSON exactly like:\n'
    '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"..."},'
    '{"id":"q02","answer":"no","evidence_quote":null}]}'
)


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


def _base_user(transcript: Dict, questions: Sequence[str]) -> str:
    return (
        "TRANSCRIPT\n"
        f"{serialize_transcript(transcript)}\n\n"
        "QUESTIONS\n"
        f"{questions_block(questions)}\n\n"
        f"{SCHEMA_HINT}"
    )


def build_l0_messages(transcript: Dict, questions: Sequence[str]) -> List[Dict]:
    """Zero-shot, single call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _base_user(transcript, questions)},
    ]


def build_l1_messages(
    transcript: Dict, questions: Sequence[str], few_shot: Sequence[Tuple[str, str]] = ()
) -> List[Dict]:
    """Few-shot, single call. ``few_shot`` is a list of (user, assistant) turns."""
    messages: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in few_shot:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": _base_user(transcript, questions)})
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
