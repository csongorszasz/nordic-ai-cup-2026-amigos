"""Conditional acoustic failures and ineligible sources keep every denominator."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from answerers.acoustic_boundaries import AcousticSkip
from benchmark_ctc import run_benchmark, validate_result


WORDS = [
    {"word": " Good.", "start": 0.0, "end": 0.5},
    {"word": " Good.", "start": 3.0, "end": 3.5},
]
ROW = {
    "question_id": "q", "transcript_id": "s", "question": "Was it good?",
    "label": 1, "answer": True, "prediction": 1, "question_type": "positive",
    "span": [3.0, 3.5], "gold": [3.1, 3.4], "duration": 5.0,
    "quote": "Good.", "word_range": [1, 1],
}


def result():
    return {
        "source_word_range": [1, 1], "span": [3.1, 3.4],
        "candidate_spans": [[3.2, 3.5], [3.1, 3.4]],
    }


@pytest.mark.parametrize("mode", ["aligned", "skip", "failure", "wrong_occurrence", "decode_failure"])
def test_every_failure_or_skip_retains_the_complete_qualified_baseline(monkeypatch, tmp_path, mode):
    missed = {**ROW, "question_id": "miss", "answer": False, "prediction": 0, "span": None, "word_range": None}
    negative = {**missed, "question_id": "negative", "label": 0, "gold": None, "question_type": "hard_negative"}
    rows = [copy.deepcopy(ROW), missed, negative]
    original = copy.deepcopy(rows)
    requests = [{"transcript_id": "s", "demonstration_tids": [], "latency_s": 1.0}]

    class Aligner:
        runtime = {"stub": True}
        calls = 0

        def load(self):
            pass

        def align(self, waveform, words, anchor, baseline):
            self.calls += 1
            assert anchor == [1, 1] and baseline == [3.2, 3.5]
            if mode == "skip":
                raise AcousticSkip("unsupported spoken form")
            if mode == "failure":
                raise RuntimeError("forced model failure")
            value = result()
            if mode == "wrong_occurrence":
                value["source_word_range"] = [0, 0]
            return value

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=1024),
    ))
    aligner = Aligner()
    status = run_benchmark(
        rows, requests, {"s": {"words": WORDS, "duration": 5.0}}, {"s": b"synthetic"},
        aligner, tmp_path, smoke=True,
        decoder=lambda _: [0.0] * (79000 if mode == "decode_failure" else 80000),
    )
    predictions = json.loads((tmp_path / "questions.json").read_text())
    report = json.loads((tmp_path / "summary.json").read_text())
    assert rows == original
    assert len(predictions) == 3 and report["positives"] == 2
    assert report["all_questions"]["candidate"]["accuracy"] == pytest.approx(2 / 3)
    assert predictions[1]["answer"] is False and predictions[1]["span"] is None
    assert predictions[2]["answer"] is False and predictions[2]["span"] is None
    assert report["deployment_qualified"] is False and report["full_corpus"] is False
    if mode == "aligned":
        assert status == 0 and predictions[0]["span"] == [3.1, 3.4]
        assert report["candidate_oracle_frozen_decision_mean_tiou"] == 0.5
    else:
        assert status == 1 and predictions[0]["span"] == [3.2, 3.5]
        assert report["candidate_oracle_frozen_decision_mean_tiou"] == pytest.approx(0.25)
        assert bool(report["operational_failures"]) is (mode != "skip")
    assert aligner.calls == (0 if mode == "decode_failure" else 1)


@pytest.mark.parametrize("change", ["duplicate", "no_baseline", "invalid", "nan", "missing_primary", "wrong_occurrence"])
def test_invalid_proposal_contract_is_not_silently_accepted(change):
    value = result()
    if change == "duplicate":
        value["candidate_spans"].append(value["candidate_spans"][0])
    elif change == "no_baseline":
        value["candidate_spans"].pop(0)
    elif change == "invalid":
        value["candidate_spans"].append([-1.0, 0.0])
    elif change == "nan":
        value["candidate_spans"].append([float("nan"), 4.0])
    elif change == "missing_primary":
        value["span"] = [4.0, 4.2]
    else:
        value["source_word_range"] = [0, 0]
    with pytest.raises(ValueError, match="Acoustic"):
        validate_result(ROW, value)


def test_proposal_acceptance_cannot_depend_on_reference_overlap():
    value = result()
    validate_result(ROW, value)
    validate_result({**ROW, "gold": [0.0, 0.1], "label": 0, "question_type": "hard_negative"}, value)
