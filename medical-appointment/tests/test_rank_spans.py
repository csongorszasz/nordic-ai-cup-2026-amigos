"""Anchored utility features and unchanged-span safety."""

import numpy as np

from rank_spans import FEATURES, Case, candidates, fit, select


def words():
    return [
        {"word": " " + token, "start": index * 0.5, "end": index * 0.5 + 0.4,
         "seg_idx": index // 4, "probability": 0.9}
        for index, token in enumerate("The dose is 100 mg daily after a meal.".split())
    ]


def test_baseline_is_always_zero_delta_and_candidate_inputs_need_no_gold():
    spans, features = candidates("Is the dose 100 mg?", words(), [0.2, 2.4], 4.5)
    assert spans[0] == [0.2, 2.4]
    assert features.shape[1] == len(FEATURES)
    assert np.array_equal(features[0], np.zeros(len(FEATURES)))
    assert np.isfinite(features).all()
    assert all(0 <= span[0] < span[1] <= 4.5 for span in spans)


def test_zero_utility_preserves_the_original_span():
    spans, features = candidates("Is the dose 100 mg?", words(), [0.2, 2.4], 4.5)
    case = Case({}, spans, features, np.zeros(len(spans)), np.zeros((24, 24)), np.zeros(24))
    assert select(case, np.zeros(24), 0.0) == 0
    assert select(case, np.ones(24), 1000.0) == 0


def test_negative_decisions_have_no_span_candidates():
    spans, features = candidates("Was the dose 200 mg?", words(), None, 4.5)
    assert spans == [None]
    assert np.array_equal(features, np.zeros((1, len(FEATURES))))


def test_ridge_targets_candidate_improvement_not_passage_overlap():
    features = np.zeros((2, len(FEATURES)))
    features[1, 0] = 1.0
    delta = np.array([0.0, 0.5])
    case = Case(
        {"label": 1}, [[0.0, 1.0], [0.2, 1.0]], features, delta,
        features.T @ features / 2, features.T @ delta / 2,
    )
    weights = fit([case], 0.01)
    assert select(case, weights, 0.01) == 1
