"""Summary of a recorded validation session."""

import json
import sys

from offline.summarize_recording import frame_coverage, main, summarize
from offline.record_dataset import ValidationDatasetRecorder
from tests.test_recorder import _request, _response


def _session(tmp_path):
    session = tmp_path / "seq"
    for name in ("images", "metadata", "responses"):
        (session / name).mkdir(parents=True)
    (session / "images" / "frame_000000.png").write_bytes(b"x")
    (session / "metadata" / "frame_000000.json").write_text(
        json.dumps({"resolution_level": 0, "camera_command_feedback": None})
    )
    (session / "responses" / "frame_000000.json").write_text(
        json.dumps({"annotations": [{"object_id": "tank"}], "requested_view": None})
    )
    (session / "index.jsonl").write_text(json.dumps({"elapsed_ms": 12.0}) + "\n")
    return session


def test_summary_reports_counts_classes_and_latency(tmp_path, capsys):
    summarize(_session(tmp_path))
    output = capsys.readouterr().out

    assert "inputs (images)   1" in output
    assert "annotations/frame" in output
    assert "tank" in output
    assert "latency" in output


def test_main_accepts_explicit_dir(tmp_path, monkeypatch, capsys):
    session = _session(tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize_recording", "--dir", str(session)])

    assert main() == 0
    assert "Sequence:" in capsys.readouterr().out


def test_coverage_reports_gaps_duplicates_and_unknown_tail_without_large_allocations():
    coverage = frame_coverage([0, 2, 2], expected_frames=4)
    assert coverage["missing_index_ranges_half_open"] == [[1, 2], [3, 4]]
    assert coverage["missing_count"] == 2
    assert coverage["duplicate_indices"] == {"2": 2}
    assert coverage["tail_coverage_known"] is True
    sparse = frame_coverage([0, 1_000_000_000])
    assert sparse["missing_count"] == 999_999_999
    assert sparse["tail_coverage_known"] is False


def test_summary_never_invents_ap_or_evaluator_acceptance(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    recorder.start("seq")
    for index in (0, 2):
        request = _request(index)
        capture = recorder.record_frame(request)
        recorder.record_response(request, _response(request), 12.0, capture=capture, diagnostics={
            "request_id": request.request_id, "events": [], "output_sources": ["fresh"],
            "raw_detections": [{"class_name": "small_launcher", "confidence": 0.002}],
        })
    recorder.shutdown()
    report = summarize(tmp_path / "seq", expected_frames=4)
    assert report["score_available"] is False
    assert report["evaluator_acceptance_known"] is False
    assert "map50" not in report
    assert report["coverage"]["missing_count"] == 2
    assert report["complete_capture"] is False
    assert report["raw_below_0_01"] == 2
    assert report["raw_confidence_median"] == 0.002


def test_complete_capture_requires_an_explicit_denominator_and_all_artifacts(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    recorder.start("seq")
    request = _request()
    capture = recorder.record_frame(request)
    recorder.record_response(request, _response(request), 1.0, capture=capture)
    recorder.shutdown()
    assert summarize(tmp_path / "seq", expected_frames=1)["complete_capture"] is True
    assert summarize(tmp_path / "seq")["complete_capture"] is False
    (tmp_path / "seq" / "responses" / f"{capture.stem}.json").unlink()
    assert summarize(tmp_path / "seq", expected_frames=1)["complete_capture"] is False


def test_matching_file_counts_do_not_hide_misattributed_responses(tmp_path):
    recorder = ValidationDatasetRecorder(tmp_path)
    recorder.start("seq")
    request = _request()
    capture = recorder.record_frame(request)
    recorder.record_response(request, _response(request), 1.0, capture=capture)
    recorder.shutdown()
    response_path = tmp_path / "seq" / "responses" / f"{capture.stem}.json"
    response = json.loads(response_path.read_text())
    response["request_id"] = "another-request"
    response_path.write_text(json.dumps(response))
    report = summarize(tmp_path / "seq", expected_frames=1)
    assert report["complete_capture"] is False
    assert any("response identity mismatch" in error for error in report["integrity_errors"])


def test_corrupt_index_lines_are_reported_not_silently_ignored(tmp_path):
    session = _session(tmp_path)
    with (session / "index.jsonl").open("a") as stream:
        stream.write("not JSON\n")
    report = summarize(session)
    assert "index.jsonl:2" in report["integrity_errors"][0]
    assert report["complete_capture"] is False
