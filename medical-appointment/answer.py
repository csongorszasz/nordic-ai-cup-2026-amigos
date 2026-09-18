"""Decide a question and localize its evidence span.

Two stages:

1. **Decide** at clause level — score every window against the question's
   proposition; the best entailment must clear ``TAU`` and survive the numeric
   guard.
2. **Localize** — search contiguous word sub-ranges inside the winning window
   ± ``NEIGHBOURS`` neighbours (ADR-0001: that neighbourhood already reaches the
   global span ceiling), re-score them, and return the tightest one that still
   entails.

The returned span is always a word range, never a whole window.
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from guards import numeric_guard
from questions import to_proposition
from verifier import nli
from windows import Window, join_words

logger = logging.getLogger(__name__)

TAU = float(os.environ.get("MEDAPP_NLI_TAU", "0.5"))
NEIGHBOURS = int(os.environ.get("MEDAPP_LOCALIZE_NEIGHBOURS", "1"))
# How many neighbour clauses to merge into the decision premise. The gold
# evidence crosses our clause boundaries ~71% of the time; merging recovers most
# of the lost positive recall (diagnose_recall: 73% -> 90% base, 75% -> 95% large).
DECISION_NEIGHBOURS = int(os.environ.get("MEDAPP_DECISION_NEIGHBOURS", "1"))
MAX_RANGE_WORDS = int(os.environ.get("MEDAPP_MAX_RANGE_WORDS", "16"))
MAX_CANDIDATES = int(os.environ.get("MEDAPP_MAX_CANDIDATES", "400"))
HYPOTHESIS_MODE = os.environ.get("MEDAPP_HYPOTHESIS", "proposition")
USE_GUARD = os.environ.get("MEDAPP_NUMERIC_GUARD", "1") != "0"
# How many top-scoring clauses to search for the span (their neighbourhoods are
# unioned). 1 = the winning clause only.
TOP_CLAUSES = int(os.environ.get("MEDAPP_TOP_CLAUSES", "1"))
CLAUSE_MARGIN = float(os.environ.get("MEDAPP_CLAUSE_MARGIN", "0.25"))
# A candidate within this much of the best score can be preferred if shorter:
# NLI entailment grows with added context, so the raw argmax overshoots.
LOCALIZE_MARGIN = float(os.environ.get("MEDAPP_LOCALIZE_MARGIN", "0.15"))
# argmax_long | argmax_short | shortest_margin | shortest_tau | longest_margin | greedy_trim
SELECT_MODE = os.environ.get("MEDAPP_SELECT", "greedy_trim")
# How much entailment may drop while trimming a span.
TRIM_MARGIN = float(os.environ.get("MEDAPP_TRIM_MARGIN", "0.1"))

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "am", "do", "does", "did",
    "have", "has", "had", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "be", "been", "being", "of", "in", "on", "at",
    "for", "to", "with", "without", "after", "before", "from", "by", "as",
    "and", "or", "not", "no", "yes", "this", "that", "these", "those", "it",
    "its", "patient", "doctor", "there", "any",
}
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def hypothesis_for(question: str) -> str:
    """The text the NLI model tests against a window."""
    if HYPOTHESIS_MODE == "raw":
        return question.strip()
    return to_proposition(question)


def _content_tokens(question: str) -> set:
    return {
        token.lower()
        for token in _WORD_RE.findall(question)
        if token.lower() not in _STOPWORDS and (len(token) >= 4 or token.isdigit())
    }


def _enumerate_candidates(
    words: List[Dict], first: int, last: int, content: set
) -> List[Tuple[int, int]]:
    """All contiguous word ranges in ``[first, last]``, content-filtered."""
    ranges: List[Tuple[int, int]] = []
    for i in range(first, last + 1):
        for j in range(i, min(last, i + MAX_RANGE_WORDS - 1) + 1):
            ranges.append((i, j))

    if content:
        kept = []
        for i, j in ranges:
            text = join_words(words, i, j).lower()
            if any(token in text for token in content):
                kept.append((i, j))
        if kept:
            ranges = kept

    return ranges


def _cap_candidates(
    ranges: List[Tuple[int, int]], limit: int = MAX_CANDIDATES
) -> List[Tuple[int, int]]:
    """Truncate to ``limit`` while keeping every span length represented.

    The old cap sorted by shortest and truncated, which structurally discarded
    the longer spans (gold spans often exceed 4 s). Here we round-robin across
    lengths so the retained set spans short and long alike.
    """
    if len(ranges) <= limit:
        return ranges

    by_length: Dict[int, List[Tuple[int, int]]] = {}
    for pair in ranges:
        by_length.setdefault(pair[1] - pair[0], []).append(pair)

    kept: List[Tuple[int, int]] = []
    lengths = sorted(by_length)
    while len(kept) < limit and any(by_length.values()):
        for length in lengths:
            bucket = by_length[length]
            if bucket and len(kept) < limit:
                kept.append(bucket.pop(0))
    return kept


def _decision_texts(windows: Sequence[Window], neighbours: int) -> List[str]:
    """Premise texts for the decision: each window joined with its neighbours."""
    if neighbours <= 0:
        return [w.text for w in windows]
    count = len(windows)
    texts = []
    for index in range(count):
        lo = max(0, index - neighbours)
        hi = min(count - 1, index + neighbours)
        texts.append(" ".join(w.text for w in windows[lo:hi + 1]))
    return texts


def answer_question(
    question: str,
    words: List[Dict],
    windows: Sequence[Window],
    tau: float = TAU,
    use_guard: bool = USE_GUARD,
    return_info: bool = False,
):
    """Answer one question and, if yes, return the evidence span.

    With ``return_info=True`` also returns a dict describing the decision:
    the winning clause, the neighbourhood searched, and the chosen word range.
    """
    info = {
        "best_clause": None,
        "clause_score": None,
        "neighbourhood": None,
        "n_candidates": 0,
        "chosen": None,
        "decided_by": None,
        "t_clause": 0.0,
        "t_localize": 0.0,
        "n_clause_pairs": 0,
        "n_candidate_pairs": 0,
        "n_trim_pairs": 0,
        "guard_ok": True,
    }

    def result(is_true, span):
        return (is_true, span, info) if return_info else (is_true, span)

    if not windows or not words:
        info["decided_by"] = "empty"
        return result(False, None)

    hypothesis = hypothesis_for(question)

    decision_texts = _decision_texts(windows, DECISION_NEIGHBOURS)
    started = time.perf_counter()
    clause_scores = nli.score(decision_texts, hypothesis)
    info["t_clause"] = time.perf_counter() - started
    info["n_clause_pairs"] = len(decision_texts)
    best = max(range(len(windows)), key=lambda index: clause_scores[index])
    info["best_clause"] = best
    info["clause_score"] = clause_scores[best]

    if clause_scores[best] < tau:
        info["decided_by"] = "below_threshold"
        return result(False, None)
    if use_guard and not numeric_guard(question, decision_texts[best]):
        info["decided_by"] = "guard"
        info["guard_ok"] = False
        return result(False, None)

    content = _content_tokens(question)
    max_clause = clause_scores[best]
    qualifying = [
        index for index in range(len(windows))
        if clause_scores[index] >= max(tau, max_clause - CLAUSE_MARGIN)
    ]
    qualifying.sort(key=lambda index: -clause_scores[index])
    qualifying = qualifying[: max(1, TOP_CLAUSES)]

    candidate_set = set()
    region_first = region_last = None
    for index in qualifying:
        lo = max(0, index - NEIGHBOURS)
        hi = min(len(windows) - 1, index + NEIGHBOURS)
        first, last = windows[lo].first_word, windows[hi].last_word
        region_first = first if region_first is None else min(region_first, first)
        region_last = last if region_last is None else max(region_last, last)
        candidate_set.update(_enumerate_candidates(words, first, last, content))

    info["neighbourhood"] = (region_first, region_last)
    candidates = _cap_candidates(sorted(candidate_set))
    info["n_candidates"] = len(candidates)

    if not candidates:
        info["decided_by"] = "clause_span"
        return result(True, windows[best].span())

    started = time.perf_counter()
    texts = [join_words(words, i, j) for i, j in candidates]
    scores = nli.score(texts, hypothesis)
    info["n_candidate_pairs"] = len(texts)

    guarded = [
        (score, i, j)
        for (i, j), text, score in zip(candidates, texts, scores)
        if not use_guard or numeric_guard(question, text)
    ]

    if not guarded:
        info["t_localize"] = time.perf_counter() - started
        info["decided_by"] = "clause_span_all_vetoed"
        return result(True, windows[best].span())

    if SELECT_MODE == "greedy_trim":
        score, i, j = max(guarded, key=lambda t: (t[0], t[2] - t[1]))
        i, j, trim_pairs = _greedy_trim(
            words, i, j, question, hypothesis, use_guard, score
        )
        info["n_trim_pairs"] = trim_pairs
    else:
        i, j = _select(guarded, tau, SELECT_MODE)

    info["t_localize"] = time.perf_counter() - started
    info["chosen"] = (i, j)
    info["decided_by"] = "candidate"
    return result(True, (words[i]["start"], words[j]["end"]))


def _greedy_trim(words, i, j, question, hypothesis, use_guard, score):
    """Peel words off both ends while entailment holds.

    Entailment grows with context, so the highest-scoring candidate is usually
    the longest; trimming walks back to the shortest span that still entails.
    Returns ``(i, j, n_pairs)`` where ``n_pairs`` counts model calls made.
    """
    pairs = 0

    def score_of(a, b):
        nonlocal pairs
        text = join_words(words, a, b)
        if a > b:
            return None
        if use_guard and not numeric_guard(question, text):
            return None
        pairs += 1
        return nli.score([text], hypothesis)[0]

    while i < j:
        candidate = score_of(i + 1, j)
        if candidate is not None and candidate >= score - TRIM_MARGIN:
            i, score = i + 1, candidate
        else:
            break
    while j > i:
        candidate = score_of(i, j - 1)
        if candidate is not None and candidate >= score - TRIM_MARGIN:
            j, score = j - 1, candidate
        else:
            break
    return i, j, pairs


def _select(guarded, tau: float, mode: str):
    """Pick one guarded candidate ``(score, i, j)`` -> ``(i, j)``."""
    max_score = max(score for score, _, _ in guarded)

    if mode == "argmax_long":
        return max(guarded, key=lambda t: (t[0], t[2] - t[1]))[1:]
    if mode == "argmax_short":
        return max(guarded, key=lambda t: (t[0], t[1] - t[2]))[1:]
    if mode == "shortest_tau":
        viable = [(i, j) for score, i, j in guarded if score >= tau]
        viable = viable or [(i, j) for _, i, j in guarded]
        return min(viable, key=lambda pair: pair[1] - pair[0])

    floor = max(tau, max_score - LOCALIZE_MARGIN)
    viable = [(i, j) for score, i, j in guarded if score >= floor]
    if mode == "longest_margin":
        return max(viable, key=lambda pair: pair[1] - pair[0])
    return min(viable, key=lambda pair: pair[1] - pair[0])


def answer_all(
    questions: Sequence[str],
    words: List[Dict],
    windows: Sequence[Window],
) -> List[Tuple[bool, Optional[Tuple[float, float]]]]:
    return [answer_question(question, words, windows) for question in questions]
