"""Simple calibration metrics independent of plotting libraries."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    if len(y_true) != len(probability):
        raise ValueError("y_true and probability must have the same length")
    return float(np.mean((probability - y_true) ** 2))


def reliability_bins(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    if len(y_true) != len(probability):
        raise ValueError("y_true and probability must have the same length")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, n_bins - 1)
    mean_predicted: list[float] = []
    observed_rate: list[float] = []
    counts: list[int] = []

    for bin_index in range(n_bins):
        mask = index == bin_index
        if not np.any(mask):
            continue
        mean_predicted.append(float(np.mean(probability[mask])))
        observed_rate.append(float(np.mean(y_true[mask])))
        counts.append(int(np.sum(mask)))

    return {
        "mean_predicted": np.asarray(mean_predicted),
        "observed_rate": np.asarray(observed_rate),
        "counts": np.asarray(counts),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = reliability_bins(y_true, probability, n_bins=n_bins)
    total = int(np.sum(bins["counts"]))
    if total == 0:
        return 0.0
    absolute_gap = np.abs(bins["observed_rate"] - bins["mean_predicted"])
    weights = bins["counts"] / total
    return float(np.sum(weights * absolute_gap))
