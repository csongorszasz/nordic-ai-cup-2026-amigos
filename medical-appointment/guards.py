"""Numeric guard: veto a yes when the question's number is absent from the evidence.

Hard negatives are near-misses — the same statement at a different dose or
duration (``100 mg`` vs ``200 mg``, ``two weeks`` vs ``six weeks``). NLI is
weakest exactly here, because the sentences are lexically almost identical, so
this cheap independent check forces ``no`` when the number the question asserts
does not occur in the candidate evidence.

Deliberately conservative: it only vetoes, never confirms. If the question
asserts no number, it passes everything.
"""

import os
import re
from typing import Set

ENABLE_NUMERIC_GUARD = os.environ.get("MEDAPP_NUMERIC_GUARD", "1") != "0"

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "half": 0.5, "once": 1, "twice": 2,
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

# Alphanumeric codes (COVID-19, HbA1c, COVID19) are identifiers, not quantities.
_CODE_RE = re.compile(r"[A-Za-z]+-?\d+")
_NUM_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)?)(?![A-Za-z])")
_WORD_RE = re.compile(r"[a-zA-Z]+")


def numbers_in(text: str) -> Set[float]:
    """Every quantity mentioned in ``text``: digits, decimals and number words."""
    cleaned = _CODE_RE.sub(" ", text or "")

    values: Set[float] = set()
    for match in _NUM_RE.finditer(cleaned):
        try:
            values.add(float(match.group(1).replace(",", ".")))
        except ValueError:
            continue
    for token in _WORD_RE.findall(cleaned.lower()):
        if token in _NUMBER_WORDS:
            values.add(float(_NUMBER_WORDS[token]))
    return values


def numeric_guard(question: str, evidence_text: str) -> bool:
    """Return True if the evidence may support the question (no veto).

    Vetoes when the question asserts a quantity that the evidence does not
    contain. Returns True when the question asserts no quantity.
    """
    if not ENABLE_NUMERIC_GUARD:
        return True

    asserted = numbers_in(question)
    if not asserted:
        return True

    available = numbers_in(evidence_text)
    return asserted.issubset(available)


def vetoed(question: str, evidence_text: str) -> bool:
    return not numeric_guard(question, evidence_text)
