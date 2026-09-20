"""Timing crossovers stay at the original occurrence and never change decisions."""

from diagnose_asr import crossover


def test_crossover_is_time_anchored_and_retains_unmatched_quotes():
    words = [
        {"word": " yes", "start": 0.0, "end": 0.5},
        {"word": " yes", "start": 10.1, "end": 10.6},
    ]
    rows = [
        {"transcript_id": "s", "answer": True, "span": [10.0, 10.5], "quote": "yes"},
        {"transcript_id": "s", "answer": True, "span": [20.0, 21.0], "quote": "missing"},
        {"transcript_id": "s", "answer": False, "span": None, "quote": None},
    ]
    result, coverage = crossover(rows, {"s": words})
    assert result[0]["span"] == [10.3, 10.6]
    assert result[1]["span"] == [20.2, 21.0]
    assert result[2]["span"] is None and result[2]["answer"] is False
    assert coverage == {"aligned": 1, "retained_source": 1}
