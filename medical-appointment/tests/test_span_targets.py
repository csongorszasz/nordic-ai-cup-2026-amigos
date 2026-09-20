"""Metric-aligned targets honor correction, uniqueness, and actual audio bounds."""

from answerers.span_targets import optimal_quote_target


def test_optimal_target_drops_barely_overlapping_extra_word():
    words = [
        {"word": " dose", "start": 0.0, "end": 0.8},
        {"word": " daily", "start": 0.8, "end": 1.6},
        {"word": " Next", "start": 1.59, "end": 2.4},
    ]
    target = optimal_quote_target(words, [0.2, 1.6], 2.4)
    assert target["quote"] == "dose daily"
    assert target["span"] == [0.2, 1.6]
    assert target["tiou"] == 1.0


def test_unique_target_does_not_choose_an_ambiguous_single_word():
    words = [
        {"word": " yes", "start": 0.0, "end": 0.5},
        {"word": " fine", "start": 0.5, "end": 1.0},
        {"word": " yes", "start": 2.0, "end": 2.5},
        {"word": " now", "start": 2.5, "end": 3.0},
    ]
    raw = optimal_quote_target(words, [2.2, 2.5], 3.0, unique=False)
    unique = optimal_quote_target(words, [2.2, 2.5], 3.0)
    assert raw["quote"] == "yes"
    assert unique["quote"] != "yes"
    assert unique["span"][1] <= 3.0


def test_unservable_short_initial_reference_is_reported_not_dropped():
    words = [
        {"word": " Good", "start": 0.0, "end": 0.18},
        {"word": " afternoon.", "start": 0.18, "end": 0.6},
        {"word": " Good", "start": 5.0, "end": 5.5},
    ]
    assert optimal_quote_target(words, [0.0, 0.16], 6.0) is None
    raw = optimal_quote_target(words, [0.0, 0.16], 6.0, unique=False)
    assert raw["quote"] == "Good" and raw["tiou"] > 0.8
