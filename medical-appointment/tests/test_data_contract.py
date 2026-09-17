"""The supplied data's invariants, and that the harness agrees with it."""

from collections import Counter

from local_evaluator import oracle
from utils import (
    AUDIO_DIRECTORY,
    audio_filename_for_transcript,
    gold_evidence,
    group_questions_by_conversation,
    load_sample_questions,
)

EXPECTED = {"positive": 195, "hard_negative": 142, "off_topic": 53}


def test_question_counts_and_types():
    rows = load_sample_questions()
    assert len(rows) == 390

    types = Counter(row["question_type"] for row in rows)
    assert dict(types) == EXPECTED


def test_labels_are_balanced():
    rows = load_sample_questions()
    yes = sum(1 for row in rows if row["label"] == "1")
    assert yes == 195
    assert yes * 2 == len(rows)


def test_evidence_present_only_on_positives():
    for row in load_sample_questions():
        gold = gold_evidence(row)
        if row["question_type"] == "positive":
            assert gold is not None, row["question_id"]
            assert gold[1] >= gold[0], row["question_id"]
        else:
            assert gold is None, row["question_id"]


def test_every_conversation_has_audio_and_ten_questions():
    groups = group_questions_by_conversation()
    assert len(groups) == 39
    for audio_filename, rows in groups:
        assert (AUDIO_DIRECTORY / audio_filename).exists()
        assert len(rows) == 10


def test_grouping_preserves_csv_order():
    rows = load_sample_questions()
    groups = group_questions_by_conversation()

    assert groups[0][0] == audio_filename_for_transcript(rows[0]["transcript_id"])

    flat = [row["question_id"] for _, group in groups for row in group]
    assert flat == [row["question_id"] for row in rows]


def test_oracle_scores_one():
    statistics = oracle()
    assert statistics.accuracy == 1.0
    assert statistics.mean_tiou == 1.0
    assert statistics.final_score == 1.0
