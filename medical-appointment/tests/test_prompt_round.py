"""Prompt development sees representative cases, not confirmation references."""

import copy
import csv
import json

import pytest

from answerers.llm_prompt import build_l1_messages, render_example
from prepare_prompt_round import frozen_demonstrations, prompt_hash, review_cases, round_split


def row(tid, qid, gold, span, question_type="positive"):
    return {
        "transcript_id": tid, "question_id": qid, "question": "Was the result normal?",
        "label": int(question_type == "positive"), "question_type": question_type,
        "answer": span is not None, "prediction": int(span is not None),
        "span": span, "gold": gold, "duration": 10.0, "quote": "The result is normal.",
    }


def test_round_groups_are_disjoint_and_do_not_use_labels_to_partition():
    rows = [row(f"s{i}", f"q{i}", [1.2, 2.0], [1.0, 2.0]) for i in range(11)]
    requests = [{"demonstration_tids": ["s0"]}]
    split = round_split(rows, requests)
    development, confirmation = set(split["development_tids"]), set(split["confirmation_tids"])
    assert not development & confirmation
    assert (development | confirmation) == {f"s{i}" for i in range(1, 11)}
    assert "s0" not in development | confirmation
    assert set(split["pilot_tids"]).issubset(development)
    assert round_split([{**entry, "gold": None, "label": 0} for entry in rows], requests) == split


def test_semantic_packet_hides_gold_and_never_reads_confirmation_cases():
    rows = [
        row("dev1", "high", [1.2, 2.0], [1.0, 2.0]),
        row("dev2", "low", [1.2, 1.4], [1.0, 3.0]),
        row("dev3", "disjoint", [6.0, 7.0], [1.0, 2.0]),
        row("held", "secret", [0.0, 0.1], [3.0, 4.0]),
    ]
    original = copy.deepcopy(rows)
    cases, counts = review_cases(rows, ["dev1", "dev2", "dev3"])
    assert rows == original
    assert {case["stratum"] for case in cases} == {"high_tiou", "low_tiou", "disjoint"}
    assert all(not ({"gold", "label", "tiou", "evidence_start", "evidence_end"} & case.keys()) for case in cases)
    assert "secret" not in {case["question_id"] for case in cases}
    assert sum(counts.values()) == 3
    rows[-1]["gold"] = [8.0, 9.0]
    rows[-1]["question"] = "A changed confirmation question."
    assert review_cases(rows, ["dev1", "dev2", "dev3"]) == (cases, counts)


def test_missing_localization_strata_is_explicit_not_a_cherry_picked_success_packet():
    with pytest.raises(ValueError, match="strata"):
        review_cases([row("dev", "q", [1.2, 2.0], [1.0, 2.0])], ["dev"])


def test_prompt_hash_binds_exact_text_and_turn_order():
    messages = [{"role": "system", "content": "Rules"}, {"role": "user", "content": "Question"}]
    assert prompt_hash(messages) != prompt_hash(list(reversed(messages)))
    assert prompt_hash(messages) != prompt_hash([*messages[:-1], {"role": "user", "content": "Question "}])


def test_historical_demonstrations_must_reproduce_the_recorded_message_hash(tmp_path):
    for directory in ("data", "annotations", "transcripts"):
        (tmp_path / directory).mkdir()
    transcript = {
        "segments": [{"id": 0, "start": 1.0, "end": 2.0, "text": "It is normal."}],
        "words": [{"word": " It", "start": 1.0, "end": 1.2},
                  {"word": " is", "start": 1.2, "end": 1.4},
                  {"word": " normal.", "start": 1.4, "end": 2.0}],
    }
    kinds = ("positive", "hard_negative", "off_topic")
    source_ids = ("sample_10", "sample_17", "sample_18")
    columns = ["question_id", "transcript_id", "question_type", "question", "evidence_start", "evidence_end"]
    with (tmp_path / "data" / "question_train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for tid, kind in zip(source_ids, kinds):
            writer.writerow({
                "question_id": tid + "_q", "transcript_id": tid, "question_type": kind,
                "question": "Was it normal?", "evidence_start": "1.0" if kind == "positive" else "",
                "evidence_end": "2.0" if kind == "positive" else "",
            })
            (tmp_path / "transcripts" / f"conversation_{tid}.dc5ba020.json").write_text(json.dumps(transcript))
    (tmp_path / "annotations" / "evidence.csv").write_text(
        "question_id,bucket,start,end\nsample_17_q,refute,1.0,2.0\n"
    )
    examples = [
        render_example(transcript, "Was it normal?", True, "It is normal.", (1.0, 2.0), "base"),
        render_example(transcript, "Was it normal?", False, None, (1.0, 2.0), "base"),
        render_example(transcript, "Was it normal?", False, None, None, "base"),
    ]
    messages = build_l1_messages(transcript, ["Is the result normal?"], examples, variant="base")
    requests = [{
        "transcript_id": "target", "demonstration_tids": list(source_ids),
        "calls": [{"prompt_sha256": prompt_hash(messages)}],
    }]
    rows = [{"transcript_id": "target", "question_id": "target_q", "question": "Is the result normal?"}]
    frozen, hashes = frozen_demonstrations(tmp_path, rows, requests, {"target": transcript})
    assert frozen["target"]["few_shot"] == examples and len(hashes) == 3
    requests[0]["calls"][0]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not reproduce"):
        frozen_demonstrations(tmp_path, rows, requests, {"target": transcript})
