"""Turn a flat word stream into phrase-level evidence windows.

A *window* is a contiguous run of words with an exact ``[start, end]`` span.
It is both the unit the verifier scores and the span we return as evidence, so
its boundaries are a scoring decision: too coarse and the temporal IoU suffers,
too fine and the premise loses the context needed to judge entailment.

The construction is: split on hard breaks (sentence punctuation, long pauses,
segment boundaries), merge fragments that are too short, then split anything
too long. Every word ends up in exactly one window.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List

# A pause between words longer than this (seconds) is treated as a break.
PAUSE_GAP = 0.5
# A window shorter than this many words, or this many seconds, is merged.
MIN_WORDS = 3
MIN_DURATION = 0.7
# A window longer than this, in words or seconds, is split.
MAX_WORDS = 18
MAX_DURATION = 6.0

_SENTENCE_END = (".", "?", "!")
_CLAUSE_END = (",", ";", ":")


@dataclass
class Window:
    id: int
    start: float
    end: float
    text: str
    first_word: int
    last_word: int

    def span(self) -> tuple:
        return (self.start, self.end)

    def to_dict(self) -> Dict:
        return asdict(self)


def _ends_sentence(text: str) -> bool:
    text = text.strip()
    return bool(text) and text.endswith(_SENTENCE_END)


def _ends_clause(text: str) -> bool:
    text = text.strip()
    return bool(text) and text.endswith(_CLAUSE_END)


def _join(words: List[Dict], first: int, last: int) -> str:
    raw = "".join(words[i]["word"] for i in range(first, last + 1))
    return " ".join(raw.split())


def _duration(words: List[Dict], idxs: List[int]) -> float:
    return words[idxs[-1]]["end"] - words[idxs[0]]["start"]


def _hard_breaks(words: List[Dict]) -> List[List[int]]:
    """Cut the word stream at sentence ends, long pauses and segment bounds."""
    atoms: List[List[int]] = []
    current: List[int] = []

    for i, word in enumerate(words):
        current.append(i)
        if i + 1 >= len(words):
            break

        nxt = words[i + 1]
        pause = nxt["start"] - word["end"]
        boundary = nxt.get("seg_idx", 0) != word.get("seg_idx", 0)

        if _ends_sentence(word["word"]) or pause > PAUSE_GAP or boundary:
            atoms.append(current)
            current = []

    if current:
        atoms.append(current)
    return [a for a in atoms if a]


def _merge_short(words: List[Dict], atoms: List[List[int]]) -> List[List[int]]:
    """Merge atoms that fall below the minimum size into a neighbour."""
    changed = True
    while changed and len(atoms) > 1:
        changed = False
        i = 0
        while i < len(atoms):
            atom = atoms[i]
            if len(atom) < MIN_WORDS or _duration(words, atom) < MIN_DURATION:
                if i > 0:
                    atoms[i - 1] = atoms[i - 1] + atom
                    del atoms[i]
                    changed = True
                    continue
                if i + 1 < len(atoms):
                    atoms[i] = atom + atoms[i + 1]
                    del atoms[i + 1]
                    changed = True
                    continue
            i += 1
    return atoms


def _best_split(words: List[Dict], idxs: List[int]) -> int:
    """Index to split at: the largest pause, nudged toward the middle."""
    mid = (len(idxs) - 1) / 2
    best_k, best_score = 0, float("-inf")
    for k in range(len(idxs) - 1):
        gap = words[idxs[k + 1]]["start"] - words[idxs[k]]["end"]
        score = gap + (0.3 if _ends_clause(words[idxs[k]]["word"]) else 0.0)
        score -= 0.02 * abs(k - mid)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _split_long(words: List[Dict], atoms: List[List[int]]) -> List[List[int]]:
    """Split atoms that exceed the maximum size, preserving order."""
    result: List[List[int]] = []
    queue = list(atoms)
    while queue:
        atom = queue.pop(0)
        too_long = len(atom) > MAX_WORDS or _duration(words, atom) > MAX_DURATION
        if too_long and len(atom) > 1:
            k = _best_split(words, atom)
            queue.insert(0, atom[: k + 1])
            queue.insert(1, atom[k + 1 :])
        else:
            result.append(atom)
    return result


def build_windows(words: List[Dict]) -> List[Window]:
    """Build phrase-level windows from the transcript's flat ``words`` list."""
    if not words:
        return []

    atoms = _hard_breaks(words)
    atoms = _merge_short(words, atoms)
    atoms = _split_long(words, atoms)

    windows: List[Window] = []
    for atom in atoms:
        windows.append(
            Window(
                id=len(windows),
                start=words[atom[0]]["start"],
                end=words[atom[-1]]["end"],
                text=_join(words, atom[0], atom[-1]),
                first_word=atom[0],
                last_word=atom[-1],
            )
        )
    return windows


def windows_from_transcript(transcript: Dict) -> List[Window]:
    """Convenience wrapper: transcript dict -> windows."""
    return build_windows(transcript.get("words", []))
