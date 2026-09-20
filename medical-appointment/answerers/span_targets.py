"""Training-only metric-aligned quote targets; reference annotations stay unchanged."""

import numpy as np

from .align import align_quote_matches
from .boundaries import adjusted_span


def word_text(words, first, last):
    return " ".join("".join(word["word"] for word in words[first:last + 1]).split())


def optimal_quote_target(words, gold, duration, *, unique=True, offsets=(0.2, 0.0)):
    """Return the best feasible target, or None if none can overlap the reference."""
    if not words or gold[1] <= gold[0]:
        raise ValueError("A positive reference and timestamped words are required.")
    starts = np.asarray([word["start"] for word in words], dtype=float)[:, None]
    ends = np.asarray([word["end"] for word in words], dtype=float)[None, :]
    proposed_starts = np.maximum(0.0, starts + offsets[0])
    proposed_ends = np.minimum(duration, ends + offsets[1])
    collapsed = proposed_ends <= proposed_starts
    actual_starts = np.where(collapsed, starts, np.round(proposed_starts, 6))
    actual_ends = np.where(collapsed, ends, np.round(proposed_ends, 6))
    first, last = np.indices((len(words), len(words)))
    valid = (
        (first <= last) & (actual_starts >= 0) & (actual_ends <= duration)
        & (actual_ends > actual_starts)
        & np.isfinite(actual_starts) & np.isfinite(actual_ends)
    )
    intersection = np.maximum(
        0.0, np.minimum(gold[1], actual_ends) - np.maximum(gold[0], actual_starts)
    )
    union = np.maximum(gold[1], actual_ends) - np.minimum(gold[0], actual_starts)
    scores = np.divide(intersection, union, out=np.zeros_like(intersection), where=valid & (union > 0))
    flat = np.flatnonzero(valid & (scores > 0))
    order = np.lexsort((
        first.ravel()[flat],
        (last - first).ravel()[flat],
        -scores.ravel()[flat],
    ))
    for flat_index in flat[order]:
        i, j = int(first.ravel()[flat_index]), int(last.ravel()[flat_index])
        quote = word_text(words, i, j)
        if not quote:
            continue
        if unique:
            matches = align_quote_matches(words, quote)
            if len(matches) != 1 or matches[0][2:] != (i, j):
                continue
        span = adjusted_span([words[i]["start"], words[j]["end"]], offsets, duration)
        return {
            "quote": quote, "first_word": i, "last_word": j,
            "span": span, "tiou": float(scores.ravel()[flat_index]),
        }
    return None
