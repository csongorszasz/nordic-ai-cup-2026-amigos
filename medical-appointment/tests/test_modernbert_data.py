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


def test_cross_negative_keeps_question_group(monkeypatch):
    rows = [
        {
            "question_id": "q1", "transcript_id": "s1",
            "question_type": "off_topic", "question": "Did they cook?",
        },
        {
            "question_id": "q2", "transcript_id": "s2",
            "question_type": "off_topic", "question": "Did they dance?",
        },
    ]
    words = [
        {"word": " hello", "start": 0.0, "end": 0.5},
        {"word": " world", "start": 0.6, "end": 1.0},
    ]

    class Retriever:
        def encode(self, texts):
            return [0] * len(texts)

        def rank(self, question, passages, top_k, passage_embeddings):
            return [(passages[0], 1.0)]

    monkeypatch.setattr(md, "load_rows", lambda: rows)
    monkeypatch.setattr(md, "load_evidence", lambda: {})
    monkeypatch.setattr(
        md, "_load_word_lists",
        lambda selected: {row["transcript_id"]: words for row in selected},
    )
    examples = md.build_examples(
        top_k=1, n_cross_negatives=1, retriever=Retriever()
    )
    cross = [
        example for example in examples
        if example["question_transcript_id"] != example["passage_transcript_id"]
    ]
    assert cross
    assert all(
        example["transcript_id"] == example["question_transcript_id"]
        for example in cross
    )
    train_examples = md.build_examples(
        top_k=1, n_cross_negatives=1, retriever=Retriever(), rows=rows[:1]
    )
    assert all(example["question_transcript_id"] == "s1" for example in train_examples)
    assert all(example["passage_transcript_id"] == "s1" for example in train_examples)


def test_transcript_loader_rejects_ambiguous_cache(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(md, "TRANSCRIPTS", tmp_path)
    (tmp_path / "conversation_s1.alpha.json").write_text("{}")
    (tmp_path / "conversation_s1.beta.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="unambiguous"):
        md.transcript_path("s1", config_hash=None)
    assert md.transcript_path("s1", config_hash="alpha").name.endswith("alpha.json")
