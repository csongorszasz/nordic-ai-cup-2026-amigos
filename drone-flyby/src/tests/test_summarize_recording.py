"""Summary of a recorded validation session."""

import json
import sys

from offline.summarize_recording import main, summarize


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
