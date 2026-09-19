"""Reward geometry, serving-compatible grounding, and invalid-output isolation."""

import json
import math

import pytest

from answerers.temporal_reward import ground_quote, interval_wasserstein, quote_reward
from utils import temporal_iou


TRANSCRIPT = {
    "duration": 12.0,
    "words": [
        {"word": " First", "start": 0.8, "end": 2.0},
        {"word": " nearby", "start": 2.8, "end": 4.0},
        {"word": " distant", "start": 8.8, "end": 10.0},
        {"word": " First", "start": 10.8, "end": 12.0},
    ],
}


def output(quote):
    return json.dumps({"evidence_quote": quote})


@pytest.mark.parametrize("first,second,expected", [
    ([1, 2], [1, 2], 0.0),
    ([1, 2], [3, 4], 2.0),
    ([0, 4], [1, 3], 0.5),
    ([0, 6], [2, 3], 1.3),
    ([1, 2], [1, 4], 1.0),
])
def test_wasserstein_exact_integral_and_symmetry(first, second, expected):
    assert interval_wasserstein(first, second) == pytest.approx(expected)
    assert interval_wasserstein(second, first) == pytest.approx(expected)


def test_reward_uses_actual_offset_span_and_official_iou():
    result = quote_reward(output("nearby"), TRANSCRIPT, [3.0, 5.0])
    assert result.grounding.span == (3.0, 4.0)
    assert result.tiou == temporal_iou([3.0, 5.0], (3.0, 4.0)) == 0.5
    assert result.wasserstein == pytest.approx(math.exp(-0.5 / 2.0))
    assert result.validity == 0.0
    assert result.total == pytest.approx(result.tiou + result.wasserstein)


def test_disjoint_predictions_receive_dense_geometric_signal():
    near = quote_reward(output("nearby"), TRANSCRIPT, [1.0, 2.0])
    far = quote_reward(output("distant"), TRANSCRIPT, [1.0, 2.0])
    assert near.tiou == far.tiou == 0.0
    assert near.wasserstein > far.wasserstein > 0.0


def test_perfect_quote_reward_is_two():
    result = quote_reward(output("nearby"), TRANSCRIPT, [3.0, 4.0])
    assert result.tiou == result.wasserstein == 1.0
    assert result.total == 2.0


@pytest.mark.parametrize("raw,reason", [
    ("not JSON", "format_invalid"),
    (output(""), "format_invalid"),
    ('{"evidence_quote": 17}', "format_invalid"),
    ('["nearby"]', "format_invalid"),
    (output("invented"), "alignment_missing"),
    (output("First"), "alignment_ambiguous"),
    (None, "format_invalid"),
])
def test_invalid_output_cannot_collect_baseline_fallback_reward(raw, reason):
    transcript = {**TRANSCRIPT, "baseline_span": [3.0, 4.0]}
    result = quote_reward(raw, transcript, [3.0, 4.0])
    assert result.grounding.reason == reason
    assert result.grounding.span is None
    assert result.tiou == result.wasserstein == 0.0
    assert result.validity == result.total == -1.0


def test_short_quote_keeps_original_when_correction_would_collapse():
    transcript = {"duration": 1.0, "words": [{"word": " Hi", "start": 0.0, "end": 0.1}]}
    assert ground_quote(output("Hi"), transcript).span == (0.0, 0.1)


def test_whole_clip_does_not_change_reward_normalization():
    tight = quote_reward(output("nearby"), TRANSCRIPT, [3.0, 4.0])
    broad = quote_reward(output("First nearby distant First"), TRANSCRIPT, [3.0, 4.0])
    assert tight.total > broad.total
    assert broad.wasserstein == pytest.approx(
        math.exp(-interval_wasserstein(broad.grounding.span, [3.0, 4.0]) / (1.0 + 1e-8))
    )


@pytest.mark.parametrize("invalid", [
    None, [], [1], [2, 1], [1, 1], [-1, 2],
    [math.nan, 2], [1, math.inf], [True, 2], ["1", 2],
])
def test_invalid_gold_fails_instead_of_silent_reward(invalid):
    with pytest.raises(ValueError, match="intervals"):
        quote_reward(output("nearby"), TRANSCRIPT, invalid)
