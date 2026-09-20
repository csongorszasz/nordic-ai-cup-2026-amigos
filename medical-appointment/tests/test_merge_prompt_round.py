"""Execution sharding cannot omit failures, alter prompts, or cherry-pick runs."""

import json

import pytest

from audit_localization import exact_score
from benchmark import write_json
from merge_prompt_round import merge_results
from probe_prompt_round import ORDER_CONTROL_EXTENSION, definition_hash, selected_tids


def fixture(root):
    manifest = {"development_tids": [f"s{i}" for i in range(4)], "confirmation_tids": ["held"]}
    source_hashes = {"pipeline": "fixed"}
    signature = {"round_sha256": "round-hash", "model": "stub", "dtype": "float32"}
    rows = []
    for tid in manifest["development_tids"]:
        for answer in (True, False):
            rows.append({
                "question_id": f"{tid}_{int(answer)}", "transcript_id": tid,
                "question": f"Question {tid} {answer}?", "duration": 10.0,
                "label": int(answer), "question_type": "positive" if answer else "hard_negative",
                "gold": [0.2, 1.0] if answer else None, "answer": answer, "prediction": int(answer),
                "span": [0.0, 1.0] if answer else None, "quote": "Some words." if answer else None,
                "word_range": [0, 1] if answer else None,
            })
    parts = []
    for index in range(2):
        path = root / f"part_{index}"
        parts.append(path)
        tids = selected_tids(manifest, "development", index, 2)
        selected = [row for row in rows if row["transcript_id"] in tids]
        summary = {
            "complete": True, "full_phase": False, "phase": "development",
            "phase_tids": manifest["development_tids"], "selected_tids": tids,
            "execution_shard": {"index": index, "count": 2},
            "signature": signature, "pipeline_source_sha256": source_hashes,
            "definition_sha256": {variant: definition_hash(variant) for variant in ("base", "claim")},
            "protocol_extension": ORDER_CONTROL_EXTENSION,
            "runtime": {"stub": True}, "cpu_threads": 16,
            "strata_definition": "qualified", "limitations": ["synthetic fixture"],
            "variants": {},
        }
        for variant in ("base", "claim"):
            records = [{
                **row, "raw_span": row["span"], "span": [0.2, 1.0] if row["answer"] else None,
                "decided_by": "yes" if row["answer"] else "no",
            } for row in selected]
            if variant == "claim":
                records = [
                    {**row, "answer": False, "prediction": 0, "span": None, "raw_span": None,
                     "decided_by": "alignment"}
                    if row["question_id"] == "s2_1" else row for row in records
                ]
            requests = [{"transcript_id": tid, "latency_s": 1.0} for tid in tids]
            summary["variants"][variant] = {
                "score": exact_score(records),
                "failed_questions": sum(row["decided_by"] == "alignment" for row in records),
            }
            write_json(path / f"{variant}_questions.json", records)
            write_json(path / f"{variant}_conversations.json", requests)
        write_json(path / "summary.json", summary)
    return rows, manifest, parts, source_hashes


def test_complete_merge_retains_every_failed_and_negative_question(tmp_path):
    rows, manifest, parts, source = fixture(tmp_path)
    report, records, calls = merge_results(rows, manifest, parts, "development", source, "round-hash")
    assert report["complete"] is True and report["full_phase"] is True
    assert report["selected_tids"] == manifest["development_tids"]
    assert report["variants"]["base"]["score"]["score"] == 1.0
    assert report["variants"]["claim"]["score"]["score"] == pytest.approx(0.8)
    assert report["variants"]["claim"]["score"]["questions"] == 8
    assert report["variants"]["claim"]["failed_questions"] == 1
    assert [row["question_id"] for row in records["claim"]] == [row["question_id"] for row in rows]
    assert len(calls["claim"]) == 4 and len(report["merged_execution_shards"]) == 2
    assert report["deployment_qualified"] is False


def test_missing_duplicate_or_partial_shards_cannot_be_a_complete_result(tmp_path):
    rows, manifest, parts, source = fixture(tmp_path)
    for selected in (parts[:1], [parts[0], parts[0]]):
        with pytest.raises(ValueError):
            merge_results(rows, manifest, selected, "development", source, "round-hash")
    path = parts[1] / "summary.json"
    data = json.loads(path.read_text())
    data["complete"] = False
    write_json(path, data)
    with pytest.raises(ValueError, match="incomplete"):
        merge_results(rows, manifest, parts, "development", source, "round-hash")


@pytest.mark.parametrize("change", ["signature", "source", "prompt", "selection", "phase", "extension"])
def test_incompatible_shards_are_rejected_before_scoring(tmp_path, change):
    rows, manifest, parts, source = fixture(tmp_path)
    path = parts[1] / "summary.json"
    data = json.loads(path.read_text())
    if change == "signature":
        data["signature"]["dtype"] = "bfloat16"
    elif change == "source":
        data["pipeline_source_sha256"]["pipeline"] = "changed"
    elif change == "prompt":
        data["definition_sha256"]["claim"] = "changed"
    elif change == "selection":
        data["selected_tids"] = ["held"]
    elif change == "phase":
        data["phase"] = "confirmation"
    else:
        data["protocol_extension"] = None
    write_json(path, data)
    with pytest.raises(ValueError, match="shard"):
        merge_results(rows, manifest, parts, "development", source, "round-hash")


@pytest.mark.parametrize("change", ["missing", "double_offset", "unreported_failure", "question"])
def test_shard_outputs_must_match_their_full_reference_and_saved_scores(tmp_path, change):
    rows, manifest, parts, source = fixture(tmp_path)
    path = parts[0] / "claim_questions.json"
    data = json.loads(path.read_text())
    if change == "missing":
        data.pop()
    elif change == "double_offset":
        data[0]["span"][0] += 0.2
    elif change == "unreported_failure":
        data[0]["decided_by"] = "generation"
    else:
        data[0]["question"] = "Changed question"
    write_json(path, data)
    with pytest.raises(ValueError):
        merge_results(rows, manifest, parts, "development", source, "round-hash")
