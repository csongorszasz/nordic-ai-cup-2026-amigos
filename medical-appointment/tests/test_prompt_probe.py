"""Prompt comparisons preserve exact controls and keep confirmation locked."""

import copy
import json

import pytest

from answerers.llm_client import StubClient
from answerers.llm_prompt import build_l1_messages
from prepare_prompt_round import prompt_hash, round_split
from probe_prompt_round import (
    definition_hash, records_for, run_variants, selected_tids, validate_confirmation, validate_inputs,
)


TRANSCRIPT = {
    "duration": 2.0,
    "segments": [{"id": 0, "start": 0.0, "end": 1.2, "text": "The result is normal."}],
    "words": [
        {"word": " " + token, "start": index * 0.3, "end": (index + 1) * 0.3}
        for index, token in enumerate("The result is normal.".split())
    ],
}
ROW = {
    "transcript_id": "s", "question_id": "yes", "question": "Was the result normal?",
    "question_type": "positive", "label": 1, "gold": [0.2, 1.2],
    "answer": True, "prediction": 1, "span": [0.0, 1.2], "duration": 2.0,
    "quote": "The result is normal.", "word_range": [0, 3],
    "proposed_span": [0.0, 1.2], "original_span": [0.0, 1.2], "calibration_applied": False,
}


class Client(StubClient):
    model_name = "stub"
    revision = "a" * 40
    last_generation = {"completion_tokens": 10}


def test_paired_variants_keep_all_questions_and_the_same_unmodified_inputs(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDAPP_SPAN_CALIBRATION", raising=False)
    negative = {
        **ROW, "question_id": "no", "question": "Was the result abnormal?",
        "question_type": "hard_negative", "label": 0, "gold": None,
        "answer": False, "prediction": 0, "span": None, "quote": None, "word_range": None,
    }
    rows = [copy.deepcopy(ROW), negative]
    before = copy.deepcopy(rows)
    client = Client([json.dumps({"answers": [
        {"id": "q01", "answer": "yes", "evidence_quote": "The result is normal."},
        {"id": "q02", "answer": "no", "evidence_quote": None},
    ]})])
    results = run_variants(
        {"s": rows}, {"s": TRANSCRIPT}, {"s": {"few_shot": [], "demonstration_tids": []}},
        ["s"], client, ["base", "v1", "v1_claim"], tmp_path, 60.0,
    )
    assert rows == before and len(client.calls) == 3
    for variant, output in results.items():
        assert output["failed_questions"] == 0
        assert output["score"]["questions"] == 2 and output["score"]["accuracy"] == 1.0
        assert output["records"][0]["span"] == [0.2, 1.2]
        assert output["records"][0]["raw_span"] == [0.0, 1.2]
        assert output["records"][0]["calibration_applied"] is True
        assert output["records"][1]["span"] is None
        assert len(json.loads((tmp_path / f"{variant}_questions.json").read_text())) == 2


def test_parse_failures_remain_in_the_score(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDAPP_SPAN_CALIBRATION", raising=False)
    results = run_variants(
        {"s": [ROW]}, {"s": TRANSCRIPT}, {"s": {"few_shot": [], "demonstration_tids": []}},
        ["s"], Client(["broken"]), ["base"], tmp_path, 60.0,
    )
    assert results["base"]["failed_questions"] == 1
    assert results["base"]["score"]["questions"] == 1
    assert results["base"]["records"][0]["answer"] is False
    assert results["base"]["records"][0]["raw_answer"] is False
    assert results["base"]["records"][0]["original_span"] is None


def test_probe_refuses_inherited_calibration_and_missing_control(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDAPP_SPAN_CALIBRATION", "unapproved.json")
    with pytest.raises(ValueError, match="inherited calibration"):
        run_variants({}, {}, {}, [], Client(["{}"]), ["base"], tmp_path, 60.0)
    with pytest.raises(ValueError, match="control first"):
        run_variants({}, {}, {}, [], Client(["{}"]), ["v1"], tmp_path, 60.0)
    with pytest.raises(ValueError, match="budget"):
        run_variants({}, {}, {}, [], Client(["{}"]), ["base"], tmp_path, float("nan"))


def test_records_do_not_retain_stale_control_generation_fields():
    changed = records_for(
        [ROW], [(False, None, {"decided_by": "no", "raw_answer": False})], 2.0,
    )[0]
    assert changed["raw_answer"] is False
    assert changed["proposed_span"] is None and changed["original_span"] is None
    assert changed["calibration_applied"] is False
    with pytest.raises(ValueError, match="omitted"):
        records_for([ROW], [], 2.0)


def test_phase_selection_cannot_move_confirmation_inputs_into_a_pilot():
    manifest = {"development_tids": ["a", "b"], "confirmation_tids": ["c"], "pilot_tids": ["a"], "feasibility_tid": "a"}
    assert selected_tids(manifest, "confirmation") == ["c"]
    for changed in ({"pilot_tids": ["c"]}, {"feasibility_tid": "c"}):
        phase = "pilot" if "pilot_tids" in changed else "feasibility"
        with pytest.raises(ValueError):
            selected_tids({**manifest, **changed}, phase)


def test_input_validation_binds_exact_control_messages_and_question_order():
    rows = [{**ROW, "transcript_id": f"s{i}", "question_id": f"q{i}"} for i in range(6)]
    transcripts = {row["transcript_id"]: TRANSCRIPT for row in rows}
    requests, frozen = [], {}
    digest = prompt_hash(build_l1_messages(TRANSCRIPT, [ROW["question"]], [], variant="base"))
    for row in rows:
        tid = row["transcript_id"]
        requests.append({"transcript_id": tid, "demonstration_tids": ["s0"], "calls": [{"prompt_sha256": digest}]})
        frozen[tid] = {
            "few_shot": [], "demonstration_tids": ["s0"],
            "question_ids": [row["question_id"]], "prompt_sha256": digest,
        }
    manifest = round_split(rows, requests)
    assert len(validate_inputs(rows, requests, transcripts, manifest, frozen)) == 6
    frozen["s1"]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="no longer reproduce"):
        validate_inputs(rows, requests, transcripts, manifest, frozen)


def test_confirmation_requires_frozen_development_and_explicit_semantic_review():
    signature = {"model": "stub", "round_sha256": "bound"}
    source = {"pipeline": "unchanged"}
    manifest = {"development_tids": ["a", "b"]}
    selection = {
        "variant": "v1_claim", "semantic_review_complete": True, "signature": signature,
        "definition_sha256": definition_hash("v1_claim"),
    }
    development = {
        "phase": "development", "complete": True, "selected_tids": ["a", "b"],
        "signature": signature, "pipeline_source_sha256": source,
        "definition_sha256": {"v1_claim": definition_hash("v1_claim")},
        "variants": {"base": {"failed_questions": 0}, "v1_claim": {"failed_questions": 0}},
    }
    assert validate_confirmation(selection, development, manifest, signature, source) == ["base", "v1_claim"]
    for changed in (
        {**selection, "semantic_review_complete": False},
        {**selection, "definition_sha256": "changed"},
        {**selection, "signature": {"model": "another"}},
    ):
        with pytest.raises(ValueError, match="Confirmation"):
            validate_confirmation(changed, development, manifest, signature, source)
