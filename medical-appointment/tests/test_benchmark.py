"""Paired score comparisons and snapshot safety."""

import json
import tarfile

import pytest

from benchmark import compare_reference, paired_comparison
from idun import run


def record(qid, tid, label, answer, span=None):
    return {
        "question_id": qid, "transcript_id": tid, "label": label,
        "question_type": "positive" if label else "hard_negative",
        "prediction": int(answer), "answer": answer,
        "gold": [1.0, 2.0] if label else None, "span": span,
    }


def test_paired_bootstrap_uses_complete_conversation_scores():
    baseline = [record("p", "a", 1, True, [1.0, 2.0]), record("n", "b", 0, False)]
    result = paired_comparison(baseline, baseline, samples=100)
    assert result["score_delta"] == 0.0
    assert result["conversation_bootstrap_95pct"] == [0.0, 0.0]
    improved = [record("p", "a", 1, False), baseline[1]]
    assert paired_comparison(improved, baseline, samples=100)["score_delta"] == pytest.approx(0.8)


def test_paired_comparison_rejects_missing_questions():
    records = [record("p", "a", 1, True, [1.0, 2.0])]
    with pytest.raises(ValueError, match="same unique"):
        paired_comparison(records, [])


def test_snapshot_excludes_captures_and_normalizes_source(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "example.py").write_bytes(b"x = 1\r\n")
    (tmp_path / "captured").mkdir()
    (tmp_path / "captured" / "private.json").write_text("{}")
    monkeypatch.setattr(
        run, "command",
        lambda args, **kwargs: "example.py\ncaptured/private.json\n" if "ls-files" in args else "commit\n",
    )
    archive = tmp_path / "source.tgz"
    manifest = run.snapshot(archive, {"script": "example.py"})
    with tarfile.open(archive) as handle:
        assert "captured/private.json" not in handle.getnames()
        assert handle.extractfile("example.py").read() == b"x = 1\n"
        assert json.load(handle.extractfile("run_request.json"))["script"] == "example.py"
    assert set(manifest["files"]) == {"example.py"}


def test_snapshot_includes_only_explicit_model_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "calibration.json").write_text('{"start_offset_s":0.2}')
    (tmp_path / "models" / "other.json").write_text("{}")
    monkeypatch.setattr(
        run, "command", lambda args, **kwargs: "" if "ls-files" in args else "commit\n"
    )
    archive = tmp_path / "source.tgz"
    run.snapshot(archive, {}, assets=["models/calibration.json"])
    with tarfile.open(archive) as handle:
        assert "models/calibration.json" in handle.getnames()
        assert "models/other.json" not in handle.getnames()


def test_public_probe_requires_the_qualified_predictions():
    expected = [
        {"answer": True, "span": [1.2, 2.0]},
        {"answer": False, "span": None},
    ]
    payload = {"answers": [True, False], "evidence_start": [1.2, None], "evidence_end": [2.0, None]}
    assert run.matches_expected(payload, expected)
    assert not run.matches_expected({**payload, "evidence_start": [1.0, None]}, expected)
    assert not run.matches_expected({**payload, "answers": [1, False]}, expected)


def test_method_comparison_requires_identical_transcripts():
    rows = [record("p", "a", 1, True, [1.0, 2.0])]
    manifest = [{
        "transcript_id": "a", "audio_sha256": "audio", "transcript_sha256": "text",
        "demonstration_tids": ["demo"],
    }]
    result = compare_reference(rows, rows, manifest, manifest)
    assert result["fixed_start_0_2"]["demonstration_disjoint"]["paired"]["score_delta"] == 0
    changed = [{**manifest[0], "transcript_sha256": "different"}]
    with pytest.raises(ValueError, match="Unmatched transcript"):
        compare_reference(rows, rows, manifest, changed)
    precision_trial = compare_reference(rows, rows, manifest, changed, allow_asr_change=True)
    assert precision_trial["comparison_axis"] == "asr"
    assert precision_trial["matched_audio"] is True
    assert precision_trial["matched_transcripts"] is False
    assert precision_trial["changed_transcript_ids"] == ["a"]
    changed_audio = [{**changed[0], "audio_sha256": "different-audio"}]
    with pytest.raises(ValueError, match="Unmatched audio"):
        compare_reference(rows, rows, manifest, changed_audio, allow_asr_change=True)
