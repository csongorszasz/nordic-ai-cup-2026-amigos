"""ModernBERT dataset labels (no model)."""

from answerers import modernbert_data as md


def test_positive_target_is_gold_span():
    row = {
        "question_id": "q1",
        "question_type": "positive",
        "evidence_start": "1.5",
        "evidence_end": "2.5",
    }
    assert md.target_span(row, {}) == (1.5, 2.5)
    assert md.bucket_label(row, {}) == "support"


def test_refute_target_from_evidence():
    row = {"question_id": "q2", "question_type": "hard_negative"}
    evidence = {"q2": {"bucket": "refute", "start": "3.0", "end": "4.0"}}
    assert md.target_span(row, evidence) == (3.0, 4.0)
    assert md.bucket_label(row, evidence) == "refute"


def test_absent_hard_negative_has_no_target():
    row = {"question_id": "q3", "question_type": "hard_negative"}
    evidence = {"q3": {"bucket": "absent", "start": "", "end": ""}}
    assert md.target_span(row, evidence) is None
    assert md.bucket_label(row, evidence) == "not_mentioned"


def test_off_topic_has_no_target():
    row = {"question_id": "q4", "question_type": "off_topic"}
    assert md.target_span(row, {}) is None
    assert md.bucket_label(row, {}) == "not_mentioned"
