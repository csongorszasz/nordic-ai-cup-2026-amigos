"""Finite local word-span candidates built only from inference-time inputs."""

import math

from .boundaries import adjusted_span


def local_span_lattice(words, anchor, baseline_span, duration, context_words=24):
    if (
        not isinstance(anchor, (list, tuple)) or len(anchor) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in anchor)
        or not 0 <= anchor[0] <= anchor[1] < len(words)
        or isinstance(context_words, bool) or not isinstance(context_words, int) or context_words < 0
    ):
        raise ValueError("A valid word anchor and nonnegative integer context are required.")
    if (
        not isinstance(baseline_span, (list, tuple)) or len(baseline_span) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (*baseline_span, duration))
        or not 0 <= baseline_span[0] < baseline_span[1] <= duration
    ):
        raise ValueError("The retained baseline must be a valid bounded interval.")
    low = max(0, anchor[0] - context_words)
    high = min(len(words) - 1, anchor[1] + context_words)
    first, last = anchor[0] - low, anchor[1] - low
    pairs = [[first, last]]
    spans = [list(baseline_span)]
    priors = [0.0]
    seen = {tuple(round(value, 6) for value in baseline_span)}
    for start in range(low, high + 1):
        for end in range(start, high + 1):
            raw = [float(words[start]["start"]), float(words[end]["end"])]
            if any(not math.isfinite(value) for value in raw) or raw[0] < 0 or raw[1] < raw[0]:
                raise ValueError("ASR word timing is invalid.")
            span = adjusted_span(raw, (0.2, 0.0), duration)
            if not 0 <= span[0] < span[1] <= duration:
                continue
            key = tuple(round(value, 6) for value in span)
            if key in seen:
                continue
            seen.add(key)
            pairs.append([start - low, end - low])
            spans.append(span)
            priors.append(-(abs(start - anchor[0]) + abs(end - anchor[1])) / 8.0)
    return {
        "first_word": low, "last_word": high, "pairs": pairs, "spans": spans,
        "priors": priors, "anchor": [first, last],
    }


def word_token_indices(offsets, sequence_ids, word_spans):
    if len(offsets) != len(sequence_ids):
        raise ValueError("Token offsets and sequence IDs differ in length.")
    mapping = []
    for start, end in word_spans:
        indices = [
            index for index, (offset, sequence) in enumerate(zip(offsets, sequence_ids))
            if sequence == 1 and offset is not None and offset[0] < end and offset[1] > start
        ]
        if not indices:
            raise ValueError("An ASR word has no passage token; do not truncate or guess its position.")
        mapping.append(indices)
    return mapping
