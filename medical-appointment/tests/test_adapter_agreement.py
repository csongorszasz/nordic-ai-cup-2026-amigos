"""Fixed OOF fusion has no learned threshold or decision changes."""

import json

import pytest

from evaluate_adapter_agreement import agreed_span, load_proposals
from llm_probe import score_records


def proposal():
    return {
        "question_id": "q", "transcript_id": "s", "label": 1, "prediction": 1,
        "answer": True, "span": [1.0, 2.0], "gold": [1.0, 2.0],
        "question_type": "positive", "duration": 3.0,
    }


def write_oof(directory, kind, rows, complete=True):
    (directory / f"{kind}_oof.json").write_text(json.dumps(rows))
    summary = {"complete_oof": complete, "questions": len(rows), kind: score_records(rows)}
    (directory / "summary.json").write_text(json.dumps(summary))


def test_agreement_keeps_baseline_when_citations_disagree():
    assert agreed_span([1.0, 3.0], [10.0, 12.0], "adapter_on_agreement") == [1.0, 3.0]
    assert agreed_span(None, [1.0, 2.0], "adapter_on_agreement") is None
    assert agreed_span([1.0, 2.0], None, "midpoint_on_agreement") == [1.0, 2.0]


def test_fixed_agreement_rules_use_only_the_two_proposals():
    assert agreed_span([1.0, 3.0], [1.2, 3.2], "adapter_on_agreement") == [1.2, 3.2]
    assert agreed_span([1.0, 3.0], [1.2, 3.2], "midpoint_on_agreement") == [1.1, 3.1]


@pytest.mark.parametrize("kind", ["adapted", "grpo"])
def test_sft_and_grpo_require_their_matching_completed_summary(tmp_path, kind):
    write_oof(tmp_path, kind, [proposal()])
    summary, rows, path = load_proposals(tmp_path, kind)
    assert summary["complete_oof"]
    assert rows == [proposal()]
    assert path.name == f"{kind}_oof.json"
    changed = {**proposal(), "span": [1.1, 2.0]}
    (tmp_path / f"{kind}_oof.json").write_text(json.dumps([changed]))
    with pytest.raises(ValueError, match="score summary"):
        load_proposals(tmp_path, kind)


def test_incomplete_or_invalid_proposals_cannot_be_fused(tmp_path):
    write_oof(tmp_path, "grpo", [proposal()], complete=False)
    with pytest.raises(ValueError, match="Incomplete"):
        load_proposals(tmp_path, "grpo")
    write_oof(tmp_path, "grpo", [{**proposal(), "span": [1.0, 4.0]}])
    with pytest.raises(ValueError, match="evidence contract"):
        load_proposals(tmp_path, "grpo")
    with pytest.raises(ValueError, match="Unsupported"):
        load_proposals(tmp_path, "other")
