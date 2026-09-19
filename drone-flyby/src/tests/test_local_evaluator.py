"""Pure helpers of the local evaluation harness."""

import local_evaluator
import pytest
import requests
from local_evaluator import _percentile, size_bin_recall

CAMERA_FRAMES = [{"frame": 0, "level": 0, "center": [1920, 1080], "region": [0, 0, 3840, 2160]}]


def test_percentile_nearest_rank():
    ordered = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(ordered, 50) == 30.0
    assert _percentile(ordered, 99) == 50.0
    assert _percentile([], 50) == 0.0


def test_size_bin_boundaries_have_no_gap(monkeypatch):
    # At L0 the transmitted scale is 960/3840 = 0.25, so a source box of side
    # 28/30/61 px becomes 7.0/7.5/15.25 px on the wire.
    boxes = [[0, 0, 28, 28], [0, 0, 30, 30], [0, 0, 61, 61]]
    annotations = [{"object_id": "tank", "bbox": box} for box in boxes]
    predictions = {0: [{"object_id": "tank", "bbox": box, "confidence": 1.0} for box in boxes]}

    monkeypatch.setattr(local_evaluator, "frame_numbers", lambda scene: [0])
    monkeypatch.setattr(local_evaluator, "load_annotations", lambda frame, scene: annotations)

    bins = size_bin_recall("scene", predictions, CAMERA_FRAMES)

    assert bins["4-7px"] == (1, 1)      # exactly 7.0 px
    assert bins["8-15px"] == (1, 1)     # 7.5 px
    assert bins[">15px"] == (1, 1)      # 15.25 px


def _replay_with_clock(monkeypatch, costs, total=7, fail_first=False):
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            assert seconds >= 0
            self.now += seconds

    clock = Clock()
    sent = []

    class Session:
        def post(self, url, json, timeout):
            sent.append((json["frame_index"], clock.now))
            clock.now += costs[min(len(sent) - 1, len(costs) - 1)]
            if fail_first and len(sent) == 1:
                raise requests.Timeout()

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {
                        "frame": json["frame"], "request_id": json["request_id"],
                        "annotations": [], "requested_view": None,
                    }
            return Response()

    monkeypatch.setattr(local_evaluator.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(local_evaluator.time, "sleep", clock.sleep)
    monkeypatch.setattr(local_evaluator.requests, "Session", Session)
    monkeypatch.setattr(local_evaluator, "frame_numbers", lambda scene: list(range(total)))
    monkeypatch.setattr(local_evaluator, "load_frame", lambda frame, scene: None)
    monkeypatch.setattr(local_evaluator, "render_view", lambda image, camera: "")
    predictions, statistics = local_evaluator.replay("unused", "scene", True, 0, False)
    return sent, predictions, statistics


def test_realtime_never_sends_a_frame_before_emission(monkeypatch):
    sent, _, statistics = _replay_with_clock(monkeypatch, [0.01])
    assert [frame for frame, _ in sent] == list(range(7))
    for frame, sent_at in sent:
        assert sent_at >= frame * local_evaluator.FRAME_INTERVAL_SECONDS - 1e-9
    assert statistics.frames_skipped == 0


def test_realtime_burst_does_not_borrow_future_frame_time(monkeypatch):
    sent, predictions, statistics = _replay_with_clock(monkeypatch, [0.01, 0.8, 0.01])
    assert [frame for frame, _ in sent] == [0, 1, 3, 4, 5, 6]
    assert 2 not in predictions
    assert statistics.frames_skipped == 1
    assert statistics.frames_sent + statistics.frames_skipped == 7


def test_realtime_timeout_counts_unanswered_and_skipped(monkeypatch):
    sent, predictions, statistics = _replay_with_clock(monkeypatch, [3.34], total=12, fail_first=True)
    assert [frame for frame, _ in sent] == [0, 10]
    assert set(predictions) == {10}
    assert statistics.timeouts == 1
    assert statistics.frames_unanswered == 1
    assert statistics.frames_sent + statistics.frames_skipped == 12


def test_score_explicit_subset_keeps_missing_frames(monkeypatch):
    annotations = [{"object_id": "tank", "bbox": [0, 0, 50, 50]}]
    monkeypatch.setattr(local_evaluator, "frame_numbers", lambda scene: [0, 1, 2])
    monkeypatch.setattr(local_evaluator, "load_annotations", lambda frame, scene: annotations)
    predictions = {0: [dict(annotations[0], confidence=1.0)]}
    perfect, classes = local_evaluator.score("scene", predictions, evaluation_frames=[0])
    partial, _ = local_evaluator.score("scene", predictions, evaluation_frames=[0, 1])
    assert perfect == pytest.approx(1.0)
    assert set(classes) == {"tank"}
    assert 0.49 < partial < 0.51
