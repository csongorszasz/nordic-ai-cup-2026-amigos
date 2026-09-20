"""Global evidence units with interpretation context separate from citation."""

import logging
import math
import re
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple

from answer import _STOPWORDS
from windows import join_words

from .boundaries import adjusted_span
from .passages import Passage, build_sentence_atoms, passages_from_ranges, validate_word_clock


logger = logging.getLogger(__name__)

SOURCE_RECIPE = {
    "version": 1,
    "max_episode_atoms": 8,
    "max_episode_words": 64,
    "max_episode_seconds": 30.0,
    "context_sentences_each_side": 1,
    "shortlist_units": 32,
    "shortlist_method": "lexical-family-round-robin-with-anchor",
    "citation_overlap_weight": 1.0,
    "context_overlap_weight": 0.5,
    "offsets_s": [0.2, 0.0],
}
FAMILIES = ("clause", "sentence", "episode")
_RETRIEVAL_STOPWORDS = _STOPWORDS - {"not", "no", "without", "after", "before"}


@dataclass
class EvidenceUnit(Passage):
    context_first_word: int
    context_last_word: int
    context_text: str
    families: Tuple[str, ...]
    timing: str = "raw"

    def resolved_span(self, duration):
        if self.timing == "qualified":
            return list(self.span())
        if self.timing != "raw":
            raise ValueError(f"Unknown evidence timing policy: {self.timing}")
        return adjusted_span(self.span(), (0.2, 0.0), duration)


def _unit(passage, families, words, sentences):
    starts = [sentence.first_word for sentence in sentences]
    first = max(0, bisect_right(starts, passage.first_word) - 2)
    last = min(len(sentences) - 1, bisect_right(starts, passage.last_word))
    context_first, context_last = sentences[first].first_word, sentences[last].last_word
    return EvidenceUnit(
        index=passage.index, first_word=passage.first_word, last_word=passage.last_word,
        start=passage.start, end=passage.end, text=passage.text,
        context_first_word=context_first, context_last_word=context_last,
        context_text=join_words(words, context_first, context_last),
        families=tuple(sorted(families)),
    )


def build_evidence_units(words, duration):
    validate_word_clock(words, duration)
    sentences = build_sentence_atoms(words)
    clauses = build_sentence_atoms(words, clauses=True)
    families = defaultdict(set)
    for name, atoms in (("sentence", sentences), ("clause", clauses)):
        for atom in atoms:
            families[atom.word_range()].add(name)
    for index, first in enumerate(sentences):
        for last in sentences[index + 1:index + SOURCE_RECIPE["max_episode_atoms"]]:
            if (
                last.last_word - first.first_word + 1 > SOURCE_RECIPE["max_episode_words"]
                or last.end - first.start > SOURCE_RECIPE["max_episode_seconds"]
            ):
                break
            families[(first.first_word, last.last_word)].add("episode")
    passages = passages_from_ranges(words, sorted(families))
    units = []
    for passage in passages:
        if passage.end <= passage.start:
            logger.warning("Skipping zero-duration evidence unit at words %s.", passage.word_range())
            continue
        units.append(_unit(passage, families[passage.word_range()], words, sentences))
    return units


def incumbent_unit(words, duration, span, anchor):
    validate_word_clock(words, duration)
    if (
        not isinstance(anchor, (list, tuple)) or len(anchor) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in anchor)
        or not 0 <= anchor[0] <= anchor[1] < len(words)
        or not isinstance(span, (list, tuple)) or len(span) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in span)
        or not 0 <= span[0] < span[1] <= duration
    ):
        raise ValueError("The incumbent requires a valid calibrated span and exact word anchor.")
    passage = Passage(
        -1, anchor[0], anchor[1], float(span[0]), float(span[1]), join_words(words, *anchor),
    )
    unit = _unit(passage, ("incumbent",), words, build_sentence_atoms(words))
    unit.timing = "qualified"
    return unit


def _tokens(text):
    return {
        token for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _RETRIEVAL_STOPWORDS and (len(token) > 1 or token.isdigit())
    }


def retrieval_scores(question, units):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Evidence retrieval requires a nonempty question.")
    query = _tokens(question)

    def similarity(text):
        content = _tokens(text)
        return 2 * len(query & content) / max(1, len(query) + len(content))

    return {
        unit.index: similarity(unit.text) + 0.5 * similarity(unit.context_text)
        for unit in units
    }


def shortlist_evidence(question, units, duration, *, incumbent=None, top_k=32, scores=None):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("The source shortlist budget must be a positive integer.")
    if len({unit.index for unit in units}) != len(units):
        raise ValueError("Evidence unit IDs must be unique before retrieval.")
    scores = retrieval_scores(question, units) if scores is None else scores
    if set(scores) != {unit.index for unit in units} or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in scores.values()
    ):
        raise ValueError("Retrieval scores must cover every unit with a finite value.")
    ordered = sorted(units, key=lambda unit: (-scores[unit.index], unit.first_word, unit.last_word))
    queues = [deque(unit for unit in ordered if family in unit.families) for family in FAMILIES]
    if incumbent is not None:
        if incumbent.timing != "qualified":
            raise ValueError("The retained incumbent must already carry its qualified correction.")
        nearby = [
            unit for unit in units
            if unit.first_word <= incumbent.last_word and unit.last_word >= incumbent.first_word
        ]
        queues.append(deque(sorted(nearby, key=lambda unit: (
            abs(unit.first_word - incumbent.first_word) + abs(unit.last_word - incumbent.last_word),
            -scores[unit.index], unit.first_word, unit.last_word,
        ))))
    selected = [incumbent] if incumbent is not None else []
    seen = {tuple(unit.resolved_span(duration)) for unit in selected}
    limit = top_k + len(selected)
    while len(selected) < limit:
        added = False
        for queue in queues:
            while queue:
                unit = queue.popleft()
                key = tuple(unit.resolved_span(duration))
                if key not in seen:
                    selected.append(unit)
                    seen.add(key)
                    added = True
                    break
            if len(selected) == limit:
                break
        if not added:
            break
    return selected
