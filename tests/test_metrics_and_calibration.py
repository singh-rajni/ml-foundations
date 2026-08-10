import numpy as np

from ml_foundations.calibration import brier_score, expected_calibration_error, reliability_bins
from ml_foundations.metrics import binary_metrics, confusion_counts, expected_incident_cost, threshold_sweep


def test_confusion_counts_and_metrics():
    y = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([1, 0, 1, 0, 1, 0])
    counts = confusion_counts(y, pred)
    assert (counts.tn, counts.fp, counts.fn, counts.tp) == (2, 1, 1, 2)
    metrics = binary_metrics(y, pred)
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 2 / 3)
    assert np.isclose(metrics["f1"], 2 / 3)


def test_expected_cost_weights_false_negatives():
    y = np.array([1, 0])
    pred = np.array([0, 1])
    assert expected_incident_cost(y, pred, 1, 20) == 21


def test_threshold_sweep_returns_requested_thresholds():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    rows = threshold_sweep(y, p, thresholds=np.array([0.3, 0.7]))
    assert [row["threshold"] for row in rows] == [0.3, 0.7]


def test_calibration_helpers():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert brier_score(y, p) < 0.1
    bins = reliability_bins(y, p, n_bins=2)
    assert bins["counts"].sum() == 4
    assert expected_calibration_error(y, p, n_bins=2) >= 0
