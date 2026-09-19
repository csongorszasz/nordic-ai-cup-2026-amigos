"""Audit-only oracles retain every denominator and reproduce qualified outputs."""

import copy

import pytest

from audit_localization import audit_records, exact_score, validate_qualified, validate_word_clock
from benchmark_alignment import baseline_prediction


WORDS = [
    {"word": " Fine.", "start": 0.0, "end": 0.5},
    {"word": " Fine.", "start": 3.0, "end": 3.5},
    {"word": " Take", "start": 4.0, "end": 4.4},
    {"word": " 100", "start": 4.4, "end": 4.7},
    {"word": " mg.", "start": 4.7, "end": 5.0},
]
ROW = {
    "question_id": "q1", "transcript_id": "s", "question": "Was it fine?",
    "label": 1, "answer": True, "prediction": 1, "question_type": "positive",
    "span": [0.0, 0.5], "gold": [3.2, 3.5], "duration": 6.0,
    "quote": "Fine.", "word_range": [0, 0],
}


def _fixture():
    missed = {
        **ROW, "question_id": "q2", "question": "Was 100 mg advised?",
        "answer": False, "prediction": 0, "span": None, "word_range": None,
        "quote": None, "gold": [4.2, 5.0],
    }
    negative = {
        **missed, "question_id": "q3", "question": "Was 200 mg advised?",
        "label": 0, "question_type": "hard_negative", "gold": None,
    }
    return [copy.deepcopy(ROW), missed, negative], [{"transcript_id": "s", "demonstration_tids": []}], {
        "s": {"words": copy.deepcopy(WORDS), "duration": 6.0},
    }


def test_audit_counts_missed_positives_in_every_oracle_and_preserves_negatives():
    rows, requests, transcripts = _fixture()
    original = copy.deepcopy(rows)
    audited, report = audit_records(rows, requests, transcripts)
    assert rows == original
    assert len(audited) == 3 and report["incumbent"]["questions"] == 3
    assert report["incumbent"]["positives"] == 2
    assert report["incumbent"]["accuracy"] == pytest.approx(2 / 3)
    assert report["incumbent"]["mean_tiou"] == 0
    assert audited[1]["answer"] is False and audited[1]["span"] is None
    assert audited[2]["audit"] is None and audited[2]["span"] is None
    oracle = report["oracles"]["corrected_global_words"]
    assert oracle["all_positive_mean_tiou"] == 1
    assert oracle["frozen_decision_mean_tiou"] == 0.5
    assert all(value["positive_denominator"] == 2 for value in report["oracles"].values())
    assert report["oracles"]["corrected_inside_quote"]["all_positive_mean_tiou"] == 0


def test_audit_retains_source_occurrences_and_absolute_oracle_indices():
    rows, requests, transcripts = _fixture()
    audited, report = audit_records(rows, requests, transcripts)
    entry = audited[0]["audit"]
    assert entry["disjoint_answered_yes"] is True
    assert entry["disjoint_gap_s"] == pytest.approx(2.7)
    assert [match["word_range"] for match in entry["identical_quote_occurrences"]] == [[0, 0], [1, 1]]
    assert entry["oracles"]["corrected_global_words"]["word_range"] == [1, 1]
    assert audited[0]["word_range"] == [0, 0]
    assert report["overlapping_error_counts"]["disjoint_gap_at_least_2s"] == 1
    assert report["reference"]["endpoint_grid_counts"]["0.02"] == 4
    assert report["serving_artifact_written"] is False


def test_duplicate_question_conflicts_are_reported_not_relabelled():
    rows, requests, transcripts = _fixture()
    rows[1]["question"] = rows[0]["question"]
    audited, report = audit_records(rows, requests, transcripts)
    groups = report["reference"]["duplicate_question_groups"]
    assert len(groups) == 1 and groups[0]["conflicting_reference"] is True
    assert audited[1]["gold"] == [4.2, 5.0]


def test_demo_exclusion_does_not_remove_full_corpus_questions():
    rows, requests, transcripts = _fixture()
    requests[0]["demonstration_tids"] = ["s"]
    audited, report = audit_records(rows, requests, transcripts)
    assert len(audited) == 3
    assert report["demo_disjoint_incumbent"]["questions"] == 0
    assert report["reference"]["positive_questions"] == 2


def _qualified():
    rows, _, _ = _fixture()
    predictions = [baseline_prediction(row) for row in rows]
    qualified = [{**row, "request_error": None} for row in predictions]
    summary = {
        **exact_score(predictions), "complete": True, "full_corpus": True,
        "aborted": False, "failed_conversations": 0, "timeouts": 0,
    }
    return predictions, qualified, summary


def test_qualified_replay_matches_at_full_precision():
    predicted, qualified, summary = _qualified()
    assert validate_qualified(predicted, qualified, summary)["score"] == pytest.approx(0.4 * 2 / 3)


@pytest.mark.parametrize("change", ["missing", "duplicate", "decision", "span", "score", "incomplete", "integer_yes"])
def test_qualified_mismatches_fail_before_oracle_computation(change):
    predicted, qualified, summary = _qualified()
    if change == "missing":
        qualified.pop()
    elif change == "duplicate":
        qualified.append(qualified[0])
    elif change == "decision":
        qualified[1]["answer"] = True
    elif change == "span":
        qualified[0]["span"] = [0.4, 0.5]
    elif change == "score":
        summary["score"] += 0.00001
    elif change == "incomplete":
        summary["complete"] = False
    else:
        qualified[0]["answer"] = 1
    with pytest.raises(ValueError, match="Qualified|qualified"):
        validate_qualified(predicted, qualified, summary)


@pytest.mark.parametrize("field,value", [
    ("start", -1.0), ("start", float("nan")), ("end", 6.1), ("end", True),
    ("word", ""),
])
def test_invalid_word_clock_fails_explicitly(field, value):
    words = copy.deepcopy(WORDS)
    words[0][field] = value
    with pytest.raises(ValueError, match="word"):
        validate_word_clock(words, 6.0)


def test_overlapping_and_zero_length_words_do_not_get_silently_dropped():
    words = [
        {"word": " First", "start": 0.0, "end": 0.3},
        {"word": " second", "start": 0.2, "end": 0.2},
    ]
    validate_word_clock(words, 1.0)
