"""Boundary correction is applied once and tiny cohorts are refused."""

import pytest

from compare_diarization import ensure_cohort, with_fixed_offset


def test_fixed_offset_applied_once():
    records = [{"span": [1.0, 2.0], "question_id": "q"}]
    once = with_fixed_offset(records)
    assert once[0]["span"] == [1.2, 2.0]
    # Applying it again would double the correction; the caller must not.
    twice = with_fixed_offset(once)
    assert twice[0]["span"] == [1.4, 2.0]


def test_fixed_offset_leaves_null_spans():
    assert with_fixed_offset([{"span": None}])[0]["span"] is None


def test_small_cohort_is_refused():
    ensure_cohort(set("abc"), 3)
    with pytest.raises(ValueError, match="uninformative"):
        ensure_cohort(set("ab"), 3)
