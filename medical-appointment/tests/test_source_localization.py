"""Source-localization geometry cannot use reference fields to retrieve evidence."""

import copy

import pytest

from answerers.evidence_units import SOURCE_RECIPE, build_evidence_units
from audit_localization import exact_score
from benchmark_alignment import baseline_prediction
from probe_source_localization import (
    build_runtime_case, geometry_summary, probe_geometry, validate_audit_replay,
)


def fixture():
    words = [
        {"word": " First.", "start": 0.0, "end": 0.5},
        {"word": " Normal.", "start": 2.0, "end": 2.5},
    ]
    row = {
        "question_id": "q", "transcript_id": "s", "question": "Was it normal?",
        "label": 1, "answer": True, "prediction": 1, "question_type": "positive",
        "span": [0.0, 0.5], "gold": [2.2, 2.5], "duration": 3.0,
        "quote": "First.", "word_range": [0, 0],
    }
    return row, {"words": words, "duration": 3.0}


def test_runtime_serialization_contains_no_reference_or_oracle_fields():
    row, transcript = fixture()
    units = build_evidence_units(transcript["words"], 3.0)
    case, _, _ = build_runtime_case(row, transcript, units)
    changed = {**row, "gold": None, "label": 0, "question_type": "hard_negative"}
    other, _, _ = build_runtime_case(changed, transcript, units)
    assert case == other
    assert set(case) == {
        "question_id", "transcript_id", "question", "baseline_answer", "baseline_span", "candidates",
    }
    assert all(not ({"gold", "label", "tiou", "question_type"} & candidate.keys()) for candidate in case["candidates"])
    assert case["candidates"][0]["resolved_span"] == [0.2, 0.5]


def test_geometry_reports_but_never_applies_the_best_reference_unit():
    row, transcript = fixture()
    runtime, diagnostic, _ = probe_geometry([row], {"s": transcript})
    summary = geometry_summary(diagnostic)
    assert runtime[0]["baseline_span"] == [0.2, 0.5]
    assert diagnostic[0]["oracles"]["shortlisted_units"]["span"] == [2.2, 2.5]
    assert summary["candidate_oracles"]["shortlisted_units"]["all_positive_mean_tiou"] == 1
    assert summary["candidate_oracles"]["incumbent"]["all_positive_mean_tiou"] == 0


def test_missed_positive_still_has_global_candidates_but_zero_frozen_decision_iou():
    row, transcript = fixture()
    missed = {**row, "answer": False, "prediction": 0, "span": None, "word_range": None, "quote": None}
    negative = {**missed, "question_id": "negative", "gold": None, "label": 0, "question_type": "off_topic"}
    runtime, diagnostic, _ = probe_geometry([missed, negative], {"s": transcript})
    assert len(runtime) == 2 and len(diagnostic) == 2
    assert runtime[0]["baseline_span"] is None and runtime[0]["baseline_answer"] is False
    assert len(runtime[0]["candidates"]) > 0
    summary = geometry_summary(diagnostic)
    assert summary["positives"] == 1
    assert summary["candidate_oracles"]["all_units"]["all_positive_mean_tiou"] == 1
    assert summary["candidate_oracles"]["all_units"]["frozen_decision_mean_tiou"] == 0
    assert diagnostic[1]["oracles"] is None


def audit_fixture():
    row, _ = fixture()
    rows = [row]
    predictions = [baseline_prediction(row)]
    summary = {"incumbent": exact_score(predictions), "diagnostic_only": True}
    recipe = {
        "version": 1, "planned_source_shortlist": SOURCE_RECIPE["shortlist_units"],
        "offsets_s": [0.2, 0.0],
    }
    return rows, copy.deepcopy(predictions), summary, recipe


def test_matching_audit_replay_is_accepted():
    rows, audited, summary, recipe = audit_fixture()
    assert validate_audit_replay(rows, audited, summary, recipe) == [baseline_prediction(row) for row in rows]


@pytest.mark.parametrize("change", ["partial", "calibration", "score", "budget", "offset"])
def test_stale_or_partial_audit_cannot_seed_the_geometry_probe(change):
    rows, audited, summary, recipe = audit_fixture()
    if change == "partial":
        audited.clear()
    elif change == "calibration":
        audited[0]["span"][0] += 0.2
    elif change == "score":
        summary["incumbent"]["score"] += 0.001
    elif change == "budget":
        recipe["planned_source_shortlist"] = 64
    else:
        recipe["offsets_s"] = [0.0, 0.0]
    with pytest.raises(ValueError):
        validate_audit_replay(rows, audited, summary, recipe)
