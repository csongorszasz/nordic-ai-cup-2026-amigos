"""Transfer forced-alignment times without changing ASR text or word indices."""

import math


def _letters_and_numbers(text):
    if not isinstance(text, str):
        raise ValueError("Alignment tokens must be text.")
    return "".join(character for character in text.casefold() if character.isalnum())


def word_unit_map(words, unit_texts):
    """Map original words to aligned units by exact normalized character order."""
    source = [_letters_and_numbers(word["word"]) for word in words]
    target = [_letters_and_numbers(text) for text in unit_texts]
    if not "".join(source) or "".join(source) != "".join(target):
        raise ValueError("Forced alignment changed the transcript's letters or numbers.")
    owners = [index for index, text in enumerate(target) for _ in text]
    mapping, offset = [], 0
    for text in source:
        mapping.append((owners[offset], owners[offset + len(text) - 1]) if text else None)
        offset += len(text)
    return mapping


def _finite_number(value):
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    )


def retime_words(words, units, duration):
    """Keep punctuation-only words at an adjacent zero-length boundary."""
    if not _finite_number(duration) or duration <= 0:
        raise ValueError("Alignment requires a finite, positive audio duration.")
    mapping = word_unit_map(words, [unit["text"] for unit in units])
    previous_end = 0.0
    for unit in units:
        start, end = unit["start_time"], unit["end_time"]
        if (
            not _finite_number(start) or not _finite_number(end)
            or not previous_end <= start <= end <= duration
        ):
            raise ValueError("Aligned timestamps must be finite, ordered, and within the audio.")
        previous_end = end
    first = next(region for region in mapping if region is not None)[0]
    boundary = float(units[first]["start_time"])
    result = []
    for word, region in zip(words, mapping):
        if region is None:
            start = end = boundary
        else:
            start = float(units[region[0]]["start_time"])
            end = float(units[region[1]]["end_time"])
            boundary = end
        result.append({**word, "start": start, "end": end})
    return result
