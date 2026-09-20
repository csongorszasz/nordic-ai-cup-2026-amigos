"""Prompt builders for the LLM ceiling probe (L0 zero-shot, L1 few-shot, L2 two-pass).

All pure functions so the prompt shape is unit-testable without a model. The
input never includes ``question_type`` (unavailable at evaluation time) and never
includes retrieved candidates (L0-L2 are transcript-only).
"""

import os
import re
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
VARIANT = os.environ.get("MEDAPP_LLM_PROMPT", "base")
# Few-shot variation axes (defaults reproduce the historical demonstrations).
FEWSHOT_QUOTE = os.environ.get("MEDAPP_LLM_FEWSHOT_QUOTE", "gold")
FEWSHOT_SELECT = os.environ.get("MEDAPP_LLM_FEWSHOT_SELECT", "first")
VARIANTS = (
    "base", "v1", "v2", "v3", "scoped", "full_context", "two_positive", "no_timestamps",
    "final_statement", "audit", "multi3", "complete", "reason", "v1_reason",
)

SCHEMA_HINT = (
    'Return JSON exactly like:\n'
    '{"answers":[{"id":"q01","answer":"yes","evidence_quote":"..."},'
    '{"id":"q02","answer":"no","evidence_quote":null}]}'
)


def system_prompt(variant: str) -> str:
    rules = {
        "base": "",
        "full_context": "",
        "two_positive": "",
        "no_timestamps": "",
        "final_statement": (
            "- If several passages consistently establish the SAME queried fact, "
            "prefer its final specific statement or confirmation over an earlier "
            "preliminary mention. Preserve the question's subject, temporal status, "
            "dose and qualifiers; do not substitute a later statement with a "
            "different meaning or an unspecific acknowledgement.\n"
        ),
        "v1": (
            "Work evidence-first: for each question, find the exact supporting span in "
            "the transcript first, and only then decide the answer.\n"
        ),
        "v2": (
            "- evidence_quote must be the SHORTEST contiguous span that fully "
            "establishes the answer; do not include surrounding context that is not "
            "needed.\n"
        ),
        "v3": (
            "- If the same fact is stated more than once, quote the occurrence whose "
            "wording most closely matches the question.\n"
        ),
        "complete": (
            "- evidence_quote must be a COMPLETE clause or sentence, never a clipped "
            "fragment: do not begin or end in the middle of a sentence.\n"
            "- When the supporting statement is a short answer, confirmation or value "
            "(for example \"Yes\", \"None\", \"138 over 83\"), include the immediately "
            "preceding question or statement it responds to, so the pair reads as "
            "self-contained evidence.\n"
            "- Include only the sentence(s) that establish the answer; do not add "
            "neighbouring sentences that merely provide context.\n"
        ),
        "reason": (
            "- For each question first write a short \"reason\": one sentence of at "
            "most 20 words naming the transcript sentence that establishes the answer. "
            "Then give \"answer\" and the verbatim \"evidence_quote\".\n"
            "- The reason is not scored and never replaces evidence_quote; the quote "
            "must still be copied verbatim from the transcript.\n"
        ),
        "v1_reason": (
            "Work evidence-first: for each question, find the exact supporting span in "
            "the transcript first, and only then decide the answer.\n"
            "- For each question write a short \"reason\" (at most 20 words) naming the "
            "transcript sentence that establishes the answer, then give the verbatim "
            "\"evidence_quote\", then \"answer\".\n"
            "- The reason is not scored and never replaces evidence_quote.\n"
        ),
        "audit": (
            "- This conversation is a clinical record reviewed for an audit. For the "
            "queried fact, the authoritative evidence is the clinician's documented "
            "conclusion, prescription, plan or confirmation, not the patient's "
            "request, report, or an earlier preliminary mention.\n"
            "- If the patient asks for or mentions something the clinician later "
            "confirms, enacts or documents, quote the clinician's final statement "
            "that establishes it. Preserve the question's subject, dose, timing and "
            "qualifiers.\n"
        ),
        "multi3": (
            "- For every yes, list up to THREE distinct contiguous quotes from the "
            "transcript that each independently establish the answer, ordered from "
            "most to least likely to be the exact supporting evidence. Give one "
            "quote if only one exists; never invent or paraphrase.\n"
        ),
        "scoped": (
            "- For each yes, include segment_start and segment_end: the exact sXX "
            "identifiers of the first and last transcript lines containing your quote. "
            "Cite the occurrence you actually used, not an unrelated restatement.\n"
            "- Quote only the words carrying the answer; surrounding lines may "
            "provide context without being included in evidence_quote.\n"
            "- On no, segment_start and segment_end are null.\n"
        ),
    }
    if variant not in rules:
        raise ValueError(f"Unknown LLM prompt variant: {variant!r}")
    prompt = SYSTEM_PROMPT.replace("Rules:\n", "Rules:\n" + rules[variant], 1)
    return prompt.replace("timestamped transcript", "transcript") if variant == "no_timestamps" else prompt


def schema_hint(variant: str) -> str:
    if variant == "v1":
        return (
            'Return JSON exactly like:\n'
            '{"answers":[{"id":"q01","evidence_quote":"...","answer":"yes"},'
            '{"id":"q02","evidence_quote":null,"answer":"no"}]}'
        )
    if variant == "scoped":
        return (
            'Return JSON exactly like:\n'
            '{"answers":[{"id":"q01","answer":"yes","segment_start":"s02",'
            '"segment_end":"s03","evidence_quote":"..."},'
            '{"id":"q02","answer":"no","segment_start":null,'
            '"segment_end":null,"evidence_quote":null}]}'
        )
    if variant == "multi3":
        return (
            'Return JSON exactly like:\n'
            '{"answers":[{"id":"q01","answer":"yes","evidence_quotes":["...","..."]},'
            '{"id":"q02","answer":"no","evidence_quotes":[]}]}'
        )
    if variant == "reason":
        return (
            'Return JSON exactly like:\n'
            '{"answers":[{"id":"q01","reason":"...","answer":"yes",'
            '"evidence_quote":"..."},{"id":"q02","reason":"...","answer":"no",'
            '"evidence_quote":null}]}'
        )
    if variant == "v1_reason":
        return (
            'Return JSON exactly like:\n'
            '{"answers":[{"id":"q01","reason":"...","evidence_quote":"...",'
            '"answer":"yes"},{"id":"q02","reason":"...","evidence_quote":null,'
            '"answer":"no"}]}'
        )
    return SCHEMA_HINT


def qid_for(index: int) -> str:
    """Stable id for the question at ``index`` (0-based) in request order."""
    return f"q{index + 1:02d}"


def few_shot_counts(variant: str) -> Tuple[int, int, int]:
    return (2, 1, 1) if variant == "two_positive" else (1, 1, 1)


def _segment_line(seg: Dict, timestamps: bool = True) -> str:
    """One transcript line; the speaker tag appears only when labelled."""
    speaker = seg.get("speaker")
    prefix = f"{speaker} " if speaker else ""
    if timestamps:
        return f"[s{seg['id']:02d} {seg['start']:.2f}-{seg['end']:.2f}] {prefix}{seg['text']}"
    return f"[s{seg['id']:02d}] {prefix}{seg['text']}"


def serialize_transcript(
    transcript: Dict, max_segments: Optional[int] = None, *, timestamps: bool = True
) -> str:
    """Segment-level lines with stable ids and timestamps."""
    segments = transcript.get("segments", [])
    if max_segments is not None:
        segments = segments[:max_segments]
    return "\n".join(_segment_line(seg, timestamps) for seg in segments)


def serialize_excerpt(
    transcript: Dict,
    start: Optional[float] = None,
    end: Optional[float] = None,
    pad: float = 6.0,
    max_segments: int = 8,
    timestamps: bool = True,
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
    return "\n".join(_segment_line(seg, timestamps) for seg in chosen)


def questions_block(questions: Sequence[str]) -> str:
    return "\n".join(
        f"{qid_for(index)}: {question}"
        for index, question in enumerate(questions)
    )


def _base_user(
    transcript: Dict, questions: Sequence[str], variant: str = VARIANT
) -> str:
    return (
        "TRANSCRIPT\n"
        f"{serialize_transcript(transcript, timestamps=variant != 'no_timestamps')}\n\n"
        "QUESTIONS\n"
        f"{questions_block(questions)}\n\n"
        f"{schema_hint(variant)}"
    )


def build_l0_messages(transcript: Dict, questions: Sequence[str]) -> List[Dict]:
    """Zero-shot, single call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _base_user(transcript, questions)},
    ]


def build_l1_messages(
    transcript: Dict, questions: Sequence[str], few_shot: Sequence[Tuple[str, str]] = (),
    variant: str = VARIANT,
) -> List[Dict]:
    """Few-shot, single call. ``few_shot`` is a list of (user, assistant) turns."""
    messages: List[Dict] = [{"role": "system", "content": system_prompt(variant)}]
    for user, assistant in few_shot:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": _base_user(transcript, questions, variant)})
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
    variant: str = VARIANT,
) -> Tuple[str, str]:
    """One few-shot turn pair from an example conversation."""
    start, end = evidence_span if evidence_span else (None, None)
    excerpt = (
        serialize_transcript(transcript)
        if variant == "full_context" and answer
        else serialize_excerpt(transcript, start, end, timestamps=variant != "no_timestamps")
    )
    user = (
        "TRANSCRIPT\n"
        f"{excerpt}\n\n"
        "QUESTIONS\n"
        f"q01: {question}\n\n"
        f"{schema_hint(variant)}"
    )
    import json

    entry = {"id": "q01"}
    if variant in ("reason", "v1_reason"):
        entry["reason"] = (
            "The quoted transcript sentence establishes the answer."
            if answer
            else "No transcript sentence establishes the queried fact."
        )
    if variant in ("v1", "v1_reason"):
        entry["evidence_quote"] = quote if answer else None
        entry["answer"] = "yes" if answer else "no"
    else:
        entry["answer"] = "yes" if answer else "no"
        entry["evidence_quote"] = quote if answer else None
    if variant == "scoped":
        selected = [
            word for word in transcript["words"]
            if evidence_span is not None
            and word["end"] > evidence_span[0] and word["start"] < evidence_span[1]
        ] if answer else []
        segments = transcript["segments"]
        entry["segment_start"] = (
            f"s{segments[selected[0]['seg_idx']]['id']:02d}" if selected else None
        )
        entry["segment_end"] = (
            f"s{segments[selected[-1]['seg_idx']]['id']:02d}" if selected else None
        )
    assistant = json.dumps({"answers": [entry]})
    return user, assistant


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "do",
    "does", "did", "has", "have", "had", "to", "of", "in", "on", "for", "and",
    "or", "at", "by", "with", "this", "that", "these", "those", "it", "its",
    "patient", "there", "any", "also", "still", "not", "no", "yes",
}


def _content_tokens(text: str) -> set:
    return {
        token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in _STOPWORDS
    }


def _question_similarity(target_tokens: Sequence[set], question: str) -> float:
    tokens = _content_tokens(question)
    if not tokens:
        return 0.0
    best = 0.0
    for target in target_tokens:
        if not target:
            continue
        union = len(tokens | target)
        best = max(best, len(tokens & target) / union if union else 0.0)
    return best


def _word_range(words: Sequence[Dict], start: float, end: float):
    indices = [i for i, word in enumerate(words) if word["end"] > start and word["start"] < end]
    return (indices[0], indices[-1]) if indices else None


def _clause_range(words: Sequence[Dict], first: int, last: int):
    from windows import PAUSE_GAP, _ends_sentence

    i = first
    while i > 0:
        previous, current = words[i - 1], words[i]
        if _ends_sentence(previous["word"]):
            break
        if current["start"] - previous["end"] > PAUSE_GAP:
            break
        if current.get("seg_idx", 0) != previous.get("seg_idx", 0):
            break
        i -= 1
    j = last
    while j + 1 < len(words):
        current, following = words[j], words[j + 1]
        if _ends_sentence(current["word"]):
            break
        if following["start"] - current["end"] > PAUSE_GAP:
            break
        if following.get("seg_idx", 0) != current.get("seg_idx", 0):
            break
        j += 1
    return i, j


def _turn_range(words: Sequence[Dict], first: int, last: int):
    seg_indices = [word.get("seg_idx", 0) for word in words]
    low, high = min(seg_indices[first], seg_indices[last]), max(seg_indices[first], seg_indices[last])
    indices = [i for i, seg in enumerate(seg_indices) if low <= seg <= high]
    return indices[0], indices[-1]


def demo_quote(words: Sequence[Dict], span: Tuple[float, float], mode: Optional[str] = None) -> str:
    """Positive demonstration quote under the selected few-shot extent mode."""
    from windows import join_words

    mode = (mode or FEWSHOT_QUOTE).lower()
    if mode == "gold":
        return text_between(words, span[0], span[1])
    word_range = _word_range(words, span[0], span[1])
    if word_range is None:
        return text_between(words, span[0], span[1])
    if mode == "clause":
        first, last = _clause_range(words, *word_range)
    elif mode == "turn":
        first, last = _turn_range(words, *word_range)
    else:
        first, last = word_range
    return join_words(words, first, last)


def _occurrence_score(
    transcript: Dict, row: Dict, pad: float = 6.0, max_segments: int = 8,
    threshold: float = 0.5,
) -> int:
    """Excerpt segments that restate the demonstration fact (lexical proxy).

    A higher count means the fact the question asks about is stated more than
    once inside the excerpt the demonstration will actually show, which is the
    signal the ``occurrence`` selection mode ranks on.
    """
    try:
        start, end = float(row["evidence_start"]), float(row["evidence_end"])
    except (KeyError, TypeError, ValueError):
        return 0
    fact = _content_tokens(text_between(transcript.get("words", []), start, end))
    if not fact:
        return 0
    chosen = [
        segment for segment in transcript.get("segments", [])
        if segment["end"] >= start - pad and segment["start"] <= end + pad
    ][:max_segments]
    hits = 0
    for segment in chosen:
        tokens = _content_tokens(segment.get("text", ""))
        if tokens and len(tokens & fact) / len(fact) >= threshold:
            hits += 1
    return hits


def select_few_shot_rows(
    rows_by_tid: Dict[str, List[Dict]],
    transcripts: Dict[str, Dict],
    evidence: Dict[str, Dict],
    exclude_tid: str,
    counts: Tuple[int, int, int] = (1, 1, 1),
    target_questions: Sequence[str] = (),
) -> List[Dict]:
    """Select demonstrations independently of the target's answer.

    ``first`` (default) reproduces the historical selection; ``similar`` picks
    the candidate whose question is most similar to the target's questions;
    ``occurrence`` picks the positive candidate whose excerpt restates the fact
    most often, so the demonstration shows a repeated-mention context.
    """
    selected = []
    used = {exclude_tid}
    tids = sorted(rows_by_tid)
    if not target_questions:
        target_questions = [row["question"] for row in rows_by_tid.get(exclude_tid, [])]
    target_tokens = [_content_tokens(question) for question in target_questions]
    similar = FEWSHOT_SELECT == "similar"
    occurrence = FEWSHOT_SELECT == "occurrence"
    for question_type, count in zip(("positive", "hard_negative", "off_topic"), counts):
        for _ in range(count):
            candidates = [
                row
                for tid in tids
                if tid not in used and tid in transcripts
                for row in rows_by_tid[tid]
                if row["question_type"] == question_type
                and (
                    question_type != "hard_negative"
                    or evidence.get(row["question_id"], {}).get("bucket") == "refute"
                )
            ]
            chosen = None
            if candidates:
                if similar:
                    chosen = max(
                        candidates,
                        key=lambda row: _question_similarity(target_tokens, row["question"]),
                    )
                elif occurrence and question_type == "positive":
                    chosen = max(
                        candidates,
                        key=lambda row: _occurrence_score(transcripts[row["transcript_id"]], row),
                    )
                else:
                    chosen = candidates[0]
            if chosen is not None:
                selected.append(chosen)
                used.add(chosen["transcript_id"])
    return selected


def build_few_shot(
    rows_by_tid: Dict[str, List[Dict]],
    transcripts: Dict[str, Dict],
    evidence: Dict[str, Dict],
    exclude_tid: str,
    counts: Optional[Tuple[int, int, int]] = None,
    variant: str = VARIANT,
) -> List[Tuple[str, str]]:
    """Balanced demonstrations from conversations other than the target."""
    examples: List[Tuple[str, str]] = []
    selected = select_few_shot_rows(
        rows_by_tid, transcripts, evidence, exclude_tid,
        few_shot_counts(variant) if counts is None else counts,
    )
    for row in selected:
        tid = row["transcript_id"]
        words = transcripts[tid]["words"]
        item = evidence.get(row["question_id"], {})
        if row["question_type"] == "positive":
            span = (float(row["evidence_start"]), float(row["evidence_end"]))
            quote = demo_quote(words, span)
            word_range = _word_range(words, span[0], span[1])
            if word_range is not None and FEWSHOT_QUOTE in ("clause", "turn"):
                first, last = (
                    _clause_range(words, *word_range)
                    if FEWSHOT_QUOTE == "clause"
                    else _turn_range(words, *word_range)
                )
                span = (words[first]["start"], words[last]["end"])
            examples.append(render_example(
                transcripts[tid], row["question"], True, quote, span, variant
            ))
        elif row["question_type"] == "hard_negative":
            span = None
            if item.get("start") and item.get("end"):
                span = (float(item["start"]), float(item["end"]))
            examples.append(render_example(
                transcripts[tid], row["question"], False, None, span, variant
            ))
        else:
            examples.append(render_example(
                transcripts[tid], row["question"], False, None, None, variant
            ))

    return examples
