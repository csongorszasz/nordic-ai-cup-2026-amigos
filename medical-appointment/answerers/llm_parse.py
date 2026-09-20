"""Parse the LLM's JSON answer block, tolerantly.

Models wrap JSON in prose or code fences, use ``true`` instead of ``"yes"``,
occasionally drop an id. We extract the first balanced JSON object, normalise the
answer, and report missing ids as ``None`` so the caller can count them rather
than crash.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Sequence

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}
logger = logging.getLogger(__name__)


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
    if not isinstance(items, list):
        logger.warning("LLM answers field is not a list.")
        return []
    return [item for item in items if isinstance(item, dict)]


def _complete_prefix(text: str) -> List[dict]:
    """Keep complete entries when generation stops inside the answers array."""
    match = re.match(r'^\s*\{\s*"answers"\s*:\s*\[', _strip_fences(text))
    if match is None:
        return []
    remaining = _strip_fences(text)[match.end():].lstrip()
    decoder = json.JSONDecoder()
    items = []
    while remaining.startswith("{"):
        try:
            item, end = decoder.raw_decode(remaining)
        except json.JSONDecodeError:
            break
        if not isinstance(item, dict):
            break
        items.append(item)
        remaining = remaining[end:].lstrip()
        if not remaining.startswith(","):
            break
        remaining = remaining[1:].lstrip()
    if items:
        logger.warning("Recovered %d complete answers from truncated JSON.", len(items))
    return items


def parse_answers(
    text: str, expected_ids: Sequence[str]
) -> Dict[str, Optional[dict]]:
    """Map question id -> ``{"answer": bool|None, "quote": str|None}``.

    Missing or malformed entries are ``None`` so the caller can record them as
    unanswered (counted wrong) rather than treating them as a no.
    """
    result: Dict[str, Optional[dict]] = {qid: None for qid in expected_ids}
    payload = extract_json(text)
    items = _complete_prefix(text) if payload is None else _as_items(payload)
    seen = set()

    for item in items:
        qid = item.get("id") or item.get("question_id")
        if qid is None:
            continue
        qid = str(qid)
        if qid not in result:
            continue
        if qid in seen:
            logger.warning("Duplicate LLM answer id %s; rejecting the ambiguous slot.", qid)
            result[qid] = None
            continue
        seen.add(qid)
        answer = _normalise_answer(item.get("answer"))
        quote = item.get("evidence_quote", item.get("quote"))
        if isinstance(quote, str):
            quote = quote.strip() or None
        else:
            quote = None
        candidate = item.get("candidate", item.get("passage_id", item.get("chunk")))
        if candidate is not None:
            candidate = str(candidate).strip() or None
        result[qid] = {"answer": answer, "quote": quote, "candidate": candidate}
        if "keep" in item:
            result[qid]["keep"] = item["keep"] is True
        for key in ("first_word", "last_word"):
            if key in item:
                result[qid][key] = item[key]
        if "segment_start" in item or "segment_end" in item:
            result[qid]["segment_start"] = item.get("segment_start")
            result[qid]["segment_end"] = item.get("segment_end")
    return result


def parse_decisions(
    text: str, expected_ids: Sequence[str]
) -> Dict[str, Optional[bool]]:
    """Pass 1 of L2: id -> answer (or ``None``), no quote required."""
    parsed = parse_answers(text, expected_ids)
    return {qid: (entry or {}).get("answer") for qid, entry in parsed.items()}
