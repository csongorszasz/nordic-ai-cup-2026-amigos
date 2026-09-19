"""Single-interval temporal grounding rewards; no model or fallback dependency."""

import math
from dataclasses import dataclass
from typing import Optional

from .align import align_quote_matches
from .base import normalize_answer
from .boundaries import adjusted_span
from .llm_parse import extract_json
from utils import temporal_iou


@dataclass(frozen=True)
class GroundedQuote:
    quote: Optional[str]
    span: Optional[tuple[float, float]]
    reason: str


@dataclass(frozen=True)
class TemporalReward:
    grounding: GroundedQuote
    tiou: float
    wasserstein: float
    validity: float

    @property
    def total(self):
        return self.tiou + self.wasserstein + self.validity


def _interval(span):
    if (
        not isinstance(span, (list, tuple)) or len(span) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in span
        )
        or not 0 <= span[0] < span[1]
    ):
        raise ValueError("Temporal rewards require finite, positive-length intervals.")
    return float(span[0]), float(span[1])


def interval_wasserstein(first, second):
    """Exact W1 between uniform distributions on two single intervals."""
    first, second = _interval(first), _interval(second)
    left, right = first[0] - second[0], first[1] - second[1]
    if left * right >= 0:
        return (abs(left) + abs(right)) / 2
    # Integrate the two triangles when the quantile difference changes sign.
    return (left * left + right * right) / (2 * (abs(left) + abs(right)))


def ground_quote(raw, transcript):
    payload = extract_json(raw) if isinstance(raw, str) else None
    quote = payload.get("evidence_quote") if isinstance(payload, dict) else None
    if not isinstance(quote, str) or not quote.strip():
        return GroundedQuote(None, None, "format_invalid")
    matches = align_quote_matches(transcript["words"], quote)
    if len(matches) != 1:
        reason = "alignment_missing" if not matches else "alignment_ambiguous"
        return GroundedQuote(quote, None, reason)
    span = adjusted_span(matches[0][:2], (0.2, 0.0), transcript["duration"])
    valid, checked = normalize_answer(
        (True, span), duration=transcript["duration"], context="temporal reward",
    )
    if not valid:
        return GroundedQuote(quote, None, "bounds_invalid")
    return GroundedQuote(quote, tuple(checked), "grounded")


def quote_reward(raw, transcript, gold):
    """TimeLens2-style reward specialized to the competition's one-span output.

    Gold is used only here, after decoding. Invalid generations receive -1,
    never the reward of the incumbent span that inference would retain.
    """
    gold = _interval(gold)
    grounding = ground_quote(raw, transcript)
    if grounding.span is None:
        return TemporalReward(grounding, 0.0, 0.0, -1.0)
    distance = interval_wasserstein(grounding.span, gold)
    return TemporalReward(
        grounding,
        temporal_iou(gold, grounding.span),
        math.exp(-distance / (gold[1] - gold[0] + 1e-8)),
        0.0,
    )
