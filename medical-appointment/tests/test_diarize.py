"""Pure diarization helpers: no torch, no model, no audio."""

import numpy as np

from answerers import diarize


def test_window_starts_stay_within_speech_and_audio():
    segments = [{"id": 0, "start": 0.0, "end": 2.0}, {"id": 1, "start": 3.0, "end": 5.0}]
    starts = diarize.window_starts(segments, duration=5.0)
    assert starts[0] == 0.0
    assert all(0.0 <= value <= 5.0 for value in starts)
    assert any(3.0 <= value <= 4.5 for value in starts)
    assert diarize.window_starts(segments, duration=0.0) == []


def test_two_means_separates_synthetic_clusters():
    left = np.array([[1.0, 0.0], [1.0, 0.05], [1.0, -0.05]])
    right = np.array([[0.0, 1.0], [0.05, 1.0], [-0.05, 1.0]])
    labels, separation = diarize.two_means(np.vstack([left, right]))
    assert separation > 1.0
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_two_means_single_point_is_safe():
    labels, separation = diarize.two_means(np.array([[1.0, 2.0]]))
    assert list(labels) == [0]
    assert separation == 0.0


def test_majority_and_median_smoothing():
    assert diarize.majority_label([1, 1, 0]) == (1, 2 / 3)
    assert diarize.majority_label([]) == (-1, 0.0)
    assert list(diarize.median_smooth([0, 1, 0, 1, 1, 1], width=3)) == [0, 0, 1, 1, 1, 1]


def test_assign_segments_by_window_majority():
    segments = [{"id": 0, "start": 0.0, "end": 2.0}, {"id": 1, "start": 3.0, "end": 5.0}]
    assignment = diarize.assign_segments(segments, [0.0, 1.0, 3.0, 4.0], [0, 0, 1, 1])
    assert assignment[0] == (0, 1.0)
    assert assignment[1] == (1, 1.0)


def test_role_naming_prefers_the_questioning_cluster():
    texts = {
        0: ["How are you feeling?", "Let me examine your chest and your heart."],
        1: ["I have pain in my chest.", "My back hurts when I move."],
    }
    doctor_cluster, margin = diarize.name_roles(texts)
    assert doctor_cluster == 0
    assert margin > 0.0


def test_attach_labels_only_clear_segments():
    transcript = {
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "hi"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "there"},
        ]
    }
    sidecar = {
        "segments": {
            "0": {"speaker": "doctor", "confidence": 1.0},
            "1": {"speaker": "mixed", "confidence": 0.4},
        }
    }
    diarize.attach(transcript, sidecar)
    assert transcript["segments"][0]["speaker"] == "doctor"
    assert "speaker" not in transcript["segments"][1]
