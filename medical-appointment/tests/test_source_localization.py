"""Source-localization geometry cannot use reference fields to retrieve evidence."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import probe_source_localization as probe
from answerers.evidence_units import SOURCE_RECIPE, build_evidence_units
from audit_localization import exact_score
from benchmark_alignment import baseline_prediction
from probe_source_localization import (
    build_runtime_case, contained_subrange_oracle, geometry_summary, probe_geometry,
    ranked_prediction, validate_audit_replay,
)


def fixture():
    words = [
        {"word": " First.", "start": 0.0, "end": 0.5},
        {"word": " Normal.", "start": 2.0, "end": 2.5},
    ]
    row = {
        "question_id": "q", "transcript_id": "s", "question": "Was it normal?",
        "label": 1, "answer": True, "prediction": 1, "question_type": "positive",
        "span": [0.0, 0.5], "gold": [2.2, 2.5], "duration": 3.0,
        "quote": "First.", "word_range": [0, 0],
    }
    return row, {"words": words, "duration": 3.0}


def test_runtime_serialization_contains_no_reference_or_oracle_fields():
    row, transcript = fixture()
    units = build_evidence_units(transcript["words"], 3.0)
    case, _, _ = build_runtime_case(row, transcript, units)
    changed = {**row, "gold": None, "label": 0, "question_type": "hard_negative"}
    other, _, _ = build_runtime_case(changed, transcript, units)
    assert case == other
    assert set(case) == {
        "question_id", "transcript_id", "question", "baseline_answer", "baseline_span", "candidates",
    }
    assert all(not ({"gold", "label", "tiou", "question_type"} & candidate.keys()) for candidate in case["candidates"])
    assert case["candidates"][0]["resolved_span"] == [0.2, 0.5]


def test_geometry_reports_but_never_applies_the_best_reference_unit():
    row, transcript = fixture()
    runtime, diagnostic, _ = probe_geometry([row], {"s": transcript})
    summary = geometry_summary(diagnostic)
    assert runtime[0]["baseline_span"] == [0.2, 0.5]
    assert diagnostic[0]["oracles"]["shortlisted_units"]["span"] == [2.2, 2.5]
    assert summary["candidate_oracles"]["shortlisted_units"]["all_positive_mean_tiou"] == 1
    assert summary["candidate_oracles"]["incumbent"]["all_positive_mean_tiou"] == 0


def test_missed_positive_still_has_global_candidates_but_zero_frozen_decision_iou():
    row, transcript = fixture()
    missed = {**row, "answer": False, "prediction": 0, "span": None, "word_range": None, "quote": None}
    negative = {**missed, "question_id": "negative", "gold": None, "label": 0, "question_type": "off_topic"}
    runtime, diagnostic, _ = probe_geometry([missed, negative], {"s": transcript})
    assert len(runtime) == 2 and len(diagnostic) == 2
    assert runtime[0]["baseline_span"] is None and runtime[0]["baseline_answer"] is False
    assert len(runtime[0]["candidates"]) > 0
    summary = geometry_summary(diagnostic)
    assert summary["positives"] == 1
    assert summary["candidate_oracles"]["all_units"]["all_positive_mean_tiou"] == 1
    assert summary["candidate_oracles"]["all_units"]["frozen_decision_mean_tiou"] == 0
    assert diagnostic[1]["oracles"] is None


def audit_fixture():
    row, _ = fixture()
    rows = [row]
    predictions = [baseline_prediction(row)]
    summary = {"incumbent": exact_score(predictions), "diagnostic_only": True}
    recipe = {
        "version": 1, "planned_source_shortlist": SOURCE_RECIPE["shortlist_units"],
        "offsets_s": [0.2, 0.0],
    }
    return rows, copy.deepcopy(predictions), summary, recipe


def test_matching_audit_replay_is_accepted():
    rows, audited, summary, recipe = audit_fixture()
    assert validate_audit_replay(rows, audited, summary, recipe) == [baseline_prediction(row) for row in rows]


@pytest.mark.parametrize("change", ["partial", "calibration", "score", "budget", "offset"])
def test_stale_or_partial_audit_cannot_seed_the_geometry_probe(change):
    rows, audited, summary, recipe = audit_fixture()
    if change == "partial":
        audited.clear()
    elif change == "calibration":
        audited[0]["span"][0] += 0.2
    elif change == "score":
        summary["incumbent"]["score"] += 0.001
    elif change == "budget":
        recipe["planned_source_shortlist"] = 64
    else:
        recipe["offsets_s"] = [0.0, 0.0]
    with pytest.raises(ValueError):
        validate_audit_replay(rows, audited, summary, recipe)


def test_subrange_oracle_never_searches_interpretation_context_or_bridges_units():
    row, transcript = fixture()
    units = build_evidence_units(transcript["words"], 3.0)
    first = next(unit for unit in units if unit.word_range() == (0, 0))
    assert "Normal." in first.context_text
    oracle = contained_subrange_oracle([first], row["gold"], transcript["words"], 3.0)
    assert oracle["tiou"] == 0
    assert oracle["requires_unimplemented_subrange_selection"] is True


def test_ranked_prediction_changes_only_a_grounded_span_and_preserves_calibration():
    row, transcript = fixture()
    pool = build_evidence_units(transcript["words"], 3.0)
    _, _, selected = build_runtime_case(row, transcript, pool)
    scores = [{"source_only": -1.0, "with_context": -1.0, "masked_source": -2.0} for _ in selected]
    baseline = ranked_prediction(row, selected, scores, "with_context")
    assert baseline["span"] == [0.2, 0.5] and baseline["source_timing"] == "qualified"
    normal = next(index for index, unit in enumerate(selected) if unit.text == "Normal.")
    scores[normal]["with_context"] = -0.1
    changed = ranked_prediction(row, selected, scores, "with_context")
    assert changed["span"] == [2.2, 2.5] and changed["quote"] == "Normal."
    assert changed["answer"] is True and changed["prediction"] == 1
    assert changed["gold"] == row["gold"] and changed["baseline_word_range"] == [0, 0]


def test_partial_source_scores_cannot_silently_change_the_candidate_order():
    row, transcript = fixture()
    _, _, selected = build_runtime_case(row, transcript, build_evidence_units(transcript["words"], 3.0))
    with pytest.raises(ValueError, match="one likelihood row"):
        ranked_prediction(row, selected, [], "with_context")


@pytest.mark.parametrize("fail,smoke", [(False, False), (True, False), (False, True)])
def test_rank_driver_retains_complete_outputs_and_distinguishes_smoke_from_quality(
    monkeypatch, tmp_path, fail, smoke,
):
    row, transcript = fixture()
    negative = {
        **row, "question_id": "negative", "answer": False, "prediction": 0,
        "label": 0, "gold": None, "span": None, "word_range": None,
        "question_type": "hard_negative",
    }

    class Scorer:
        last_metrics = {"cached_direct_max_abs_delta": 0.0}
        runtime = {"stub": True}

        def load(self):
            pass

        def score(self, cases, words, *, check_direct):
            assert check_direct is smoke
            if fail:
                raise RuntimeError("forced source model failure")
            return [
                [{"source_only": -1.0, "with_context": -1.0, "masked_source": -2.0} for _ in units]
                for _, units in cases
            ]

    monkeypatch.setattr(probe, "InverseQuestionScorer", Scorer)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=2048),
    ))
    status = probe.run_ranking([row, negative], {"s": transcript}, [], tmp_path, smoke=smoke)
    report = json.loads((tmp_path / "summary.json").read_text())
    assert status == int(fail)
    assert report["likelihood_feasibility_passed"] is (not fail)
    assert report["deployment_qualified"] is False
    assert report["smoke_only"] is smoke
    assert bool(report["comparisons"]) is (not smoke)
    for policy in probe.POLICIES:
        predictions = json.loads((tmp_path / f"{policy}_questions.json").read_text())
        assert len(predictions) == 2
        assert predictions[0]["span"] == [0.2, 0.5]
        assert predictions[1]["answer"] is False and predictions[1]["span"] is None
        if fail:
            assert predictions[0]["source_reason"] == "conversation_error_keep"


@pytest.mark.parametrize("custom_model", [False, True])
def test_cache_verification_requests_only_the_explicit_safe_snapshot_subset(monkeypatch, tmp_path, custom_model):
    expected_model = "facebook/wav2vec2-base-960h" if custom_model else probe.MODEL
    expected_revision = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156" if custom_model else probe.REVISION
    options = {"model": expected_model, "revision": expected_revision} if custom_model else {}
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_bytes(b"{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "model": expected_model, "revision": expected_revision, "complete": True,
        "inference_network_access": False, "path": str(snapshot),
        "files": [{"path": "config.json", "bytes": 2}, {"path": "model.safetensors", "bytes": 7}],
    }))

    def snapshot_download(model, *, revision, local_files_only, allow_patterns):
        assert model == expected_model and revision == expected_revision
        assert local_files_only is True
        assert allow_patterns == ["config.json", "model.safetensors"]
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    proof = probe.verify_model_cache(manifest, **options)
    assert proof["weights_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert proof["model"] == expected_model and proof["revision"] == expected_revision
    (snapshot / "model.safetensors").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="cache changed"):
        probe.verify_model_cache(manifest, **options)
