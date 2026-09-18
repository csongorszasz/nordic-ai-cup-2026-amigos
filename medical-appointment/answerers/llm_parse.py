"""Parse the LLM's JSON answer block, tolerantly.

Models wrap JSON in prose or code fences, use ``true`` instead of ``"yes"``,
occasionally drop an id. We extract the first balanced JSON object, normalise the
answer, and report missing ids as ``None`` so the caller can count them rather
than crash.

A reply cut off by the token or time limit never closes its outer object, so
``extract_json`` finds nothing; ``salvage_objects`` then recovers every answer
object that *did* close, so a truncation costs the tail, not all ten.
"""

import json
from typing import Dict, List, Optional, Sequence

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def extract_json(text: str):
    """Return the first balanced JSON value in ``text``, or ``None``."""
    if not text:
        return None
    cleaned = _strip_fences(text)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:index + 1])
                    except Exception:
                        break
    return None


def salvage_objects(text: str) -> List[dict]:
    """Every balanced ``{...}`` in ``text`` that parses to a dict with an id.

    Scans innermost-first: an object that fails to parse (because it contains a
    nested one, or was cut off) is skipped and its inner objects are still tried.
    """
    if not text:
        return []
    found: List[dict] = []
    stack: List[int] = []
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            try:
                value = json.loads(text[start:index + 1])
            except Exception:
                continue
            if isinstance(value, dict) and ("id" in value or "question_id" in value):
                found.append(value)
    return found


def _normalise_answer(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in _YES:
        return True
    if token in _NO:
        return False
    return None


def _as_items(payload) -> List[dict]:
    if isinstance(payload, dict):
        items = payload.get("answers")
        if items is None:
            # A single-answer object.
            items = [payload] if "answer" in payload else []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _segment_index(value) -> Optional[int]:
    """``"s07"`` / ``"7"`` / ``7`` -> ``7``; anything else -> ``None``."""
    token = str(value).strip().lower().lstrip("[").rstrip("]")
    if token.startswith("s"):
        token = token[1:]
    try:
        return int(token)
    except ValueError:
        return None


def parse_answers(
    text: str, expected_ids: Sequence[str]
) -> Dict[str, Optional[dict]]:
    """Map question id -> ``{"answer": bool|None, "quote": str|None}``.

    Missing or malformed entries are ``None`` so the caller can record them as
    unanswered (counted wrong) rather than treating them as a no.
    """
    result: Dict[str, Optional[dict]] = {qid: None for qid in expected_ids}
    payload = extract_json(text)
    items = _as_items(payload) if payload is not None else []
    if not items:
        items = salvage_objects(_strip_fences(text or ""))

    for item in items:
        qid = item.get("id") or item.get("question_id")
        if qid is None:
            continue
        qid = str(qid)
        if qid not in result:
            continue
        answer = _normalise_answer(item.get("answer"))
        quote = item.get("evidence_quote", item.get("quote"))
        if isinstance(quote, str):
            quote = quote.strip() or None
        else:
            quote = None
        candidate = item.get("candidate", item.get("passage_id", item.get("chunk")))
        if candidate is not None:
            candidate = str(candidate).strip() or None
        segment = item.get("segment", item.get("segment_id"))
        if segment is not None:
            segment = _segment_index(segment)
        result[qid] = {
            "answer": answer, "quote": quote, "candidate": candidate,
            "segment": segment,
        }
    return result


def parse_decisions(
    text: str, expected_ids: Sequence[str]
) -> Dict[str, Optional[bool]]:
    """Pass 1 of L2: id -> answer (or ``None``), no quote required."""
    parsed = parse_answers(text, expected_ids)
    return {qid: (entry or {}).get("answer") for qid, entry in parsed.items()}
