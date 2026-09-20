"""OOF coverage cannot silently omit questions or alter base decisions."""

import pytest

from train_quote_oof import validate_coverage


ROW = {
    "question_id": "q", "transcript_id": "s", "label": 1,
    "prediction": 1, "answer": True, "gold": [1.0, 2.0], "span": [0.8, 2.1],
}


def test_oof_allows_only_evidence_changes():
    validate_coverage([{**ROW, "span": [1.0, 2.0]}], [ROW])
    with pytest.raises(ValueError, match="frozen decision"):
        validate_coverage([{**ROW, "answer": False, "prediction": 0}], [ROW])


def test_oof_rejects_missing_or_duplicate_questions():
    with pytest.raises(ValueError, match="missing or repeated"):
        validate_coverage([], [ROW])
    with pytest.raises(ValueError, match="missing or repeated"):
        validate_coverage([ROW, ROW], [ROW])
