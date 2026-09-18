import numpy as np

from my_jev.metrics import expected_calibration_error, multiclass_metrics


def test_perfect_predictions_have_perfect_metrics():
    metrics = multiclass_metrics(
        [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ],
        [0, 1],
    )
    assert metrics.accuracy == 1.0
    assert metrics.brier < 1e-12
    assert metrics.nll < 1e-8
    assert metrics.ece < 1e-12


def test_ece_detects_overconfidence():
    confidence = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([1.0, 0.0, 0.0, 0.0])
    assert expected_calibration_error(confidence, correct) > 0.5
