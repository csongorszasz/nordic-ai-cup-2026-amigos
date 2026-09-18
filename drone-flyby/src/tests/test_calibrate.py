"""AP math used by the per-class calibrator."""

from offline.calibrate import average_precision

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (100.0, 100.0, 110.0, 110.0)


def test_perfect_prediction_scores_one():
    assert average_precision([(0, BOX_A, 0.9)], {0: [BOX_A]}) == 1.0


def test_no_predictions_scores_zero():
    assert average_precision([], {0: [BOX_A]}) == 0.0


def test_missing_one_of_two_ground_truths_halves_ap():
    ap = average_precision([(0, BOX_A, 0.9)], {0: [BOX_A, BOX_B]})
    assert ap == 0.5


def test_non_overlapping_prediction_scores_zero():
    assert average_precision([(0, (500.0, 500.0, 510.0, 510.0), 0.9)], {0: [BOX_A]}) == 0.0
