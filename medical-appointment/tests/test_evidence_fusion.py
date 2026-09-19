"""Fusion never changes base decisions or bridges unrelated occurrences."""

from calibrate_fusion import Policy, fit_policy, fuse


def record(span, answer=True):
    return {"answer": answer, "span": span, "duration": 100.0}


def test_fusion_preserves_no_and_missing_alternative():
    assert fuse(record(None, False), record([1.0, 2.0]), Policy("later")) is None
    assert fuse(record([1.0, 2.0]), record(None, False), Policy("later")) == [1.2, 2.0]


def test_blending_requires_overlap_and_stays_between_endpoints():
    base, alt = record([1.0, 3.0]), record([1.5, 3.5])
    assert fuse(base, alt, Policy("blend", 0.25, 0.5)) == [1.45, 3.25]
    assert fuse(base, record([20.0, 22.0]), Policy("blend", 0.25, 0.5)) == [1.2, 3.0]


def test_occurrence_preference_applies_only_to_disjoint_spans():
    base = record([1.0, 3.0])
    assert fuse(base, record([20.0, 22.0]), Policy("later")) == [20.2, 22.0]
    assert fuse(base, record([1.5, 3.5]), Policy("later")) == [1.2, 3.0]


def test_equal_utility_prefers_unchanged_policy():
    rows = [{"question_id": "q", "label": 1, "gold": [1.2, 3.0], **record([1.0, 3.0])}]
    assert fit_policy(rows, {"q": record([1.0, 3.0])}).mode == "keep"
