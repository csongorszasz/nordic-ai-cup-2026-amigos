"""Window construction invariants. Pure functions, no models or audio."""

from windows import (
    MAX_DURATION,
    MAX_WORDS,
    MIN_WORDS,
    build_windows,
    windows_from_transcript,
)


def make_words(tokens, seg=0, start=0.0, dur=0.4, gap=0.0):
    words = []
    t = start
    for token in tokens:
        words.append(
            {
                "word": " " + token,
                "start": round(t, 4),
                "end": round(t + dur, 4),
                "probability": 1.0,
                "seg_idx": seg,
            }
        )
        t += dur + gap
    return words


def expected_text(words, first, last):
    raw = "".join(words[i]["word"] for i in range(first, last + 1))
    return " ".join(raw.split())


def assert_invariants(words, wins):
    if not words:
        assert wins == []
        return

    # Sequential ids, valid spans, correct text.
    for i, win in enumerate(wins):
        assert win.id == i
        assert win.start == words[win.first_word]["start"]
        assert win.end == words[win.last_word]["end"]
        assert win.start <= win.end
        assert win.text == expected_text(words, win.first_word, win.last_word)

    # Ordered, non-overlapping.
    for a, b in zip(wins, wins[1:]):
        assert a.end <= b.start + 1e-9

    # Every word covered exactly once, contiguously.
    assert wins[0].first_word == 0
    assert wins[-1].last_word == len(words) - 1
    for a, b in zip(wins, wins[1:]):
        assert b.first_word == a.last_word + 1


def test_empty():
    assert build_windows([]) == []


def test_single_word():
    words = make_words(["Hello."])
    wins = build_windows(words)
    assert_invariants(words, wins)
    assert len(wins) == 1
    assert wins[0].text == "Hello."


def test_sentence_punctuation_splits():
    words = make_words(["Good", "morning", "everyone.", "The", "dose", "is", "high."])
    wins = build_windows(words)
    assert_invariants(words, wins)
    assert [w.text for w in wins] == ["Good morning everyone.", "The dose is high."]


def test_pause_splits():
    first = make_words(["One", "two", "three"])
    second = make_words(["four", "five", "six"], start=first[-1]["end"] + 1.0)
    words = first + second
    wins = build_windows(words)
    assert_invariants(words, wins)
    assert [w.text for w in wins] == ["One two three", "four five six"]


def test_segment_boundary_splits():
    first = make_words(["Alpha", "beta", "gamma"], seg=0)
    second = make_words(["delta", "epsilon", "zeta"], seg=1, start=first[-1]["end"])
    words = first + second
    wins = build_windows(words)
    assert_invariants(words, wins)
    assert len(wins) == 2


def test_short_fragment_merges():
    words = make_words(["Yes.", "The", "dose", "is", "low."])
    wins = build_windows(words)
    assert_invariants(words, wins)
    assert len(wins) == 1
    assert wins[0].text == "Yes. The dose is low."


def test_long_window_splits():
    words = make_words([f"w{i}" for i in range(30)])
    wins = build_windows(words)
    assert_invariants(words, wins)
    assert len(wins) > 1
    for win in wins:
        assert win.last_word - win.first_word + 1 <= MAX_WORDS
        assert win.end - win.start <= MAX_DURATION + 1e-6


def test_long_duration_splits():
    words = make_words([f"w{i}" for i in range(8)], dur=1.0)
    wins = build_windows(words)
    assert_invariants(words, wins)
    for win in wins:
        assert win.end - win.start <= MAX_DURATION + 1e-6


def test_invariants_hold_on_mixed_input():
    tokens = (
        ["Dr.", "Skov", "is", "here."]
        + ["The", "dose", "is", "100", "mg", "daily,", "after", "a", "meal."]
        + ["Thank", "you."]
    )
    words = make_words(tokens)
    wins = build_windows(words)
    assert_invariants(words, wins)
    assert all(w.last_word - w.first_word + 1 >= 1 for w in wins)


def test_windows_from_transcript():
    words = make_words(["One", "two", "three", "four."])
    wins = windows_from_transcript({"words": words})
    assert wins == build_windows(words)
