"""Classification metric helpers for binary labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    tn: int
    fp: int
    fn: int
    tp: int


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionCounts:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not set(np.unique(y_true)).issubset({0, 1}) or not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("binary labels must contain only 0 and 1")
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return ConfusionCounts(tn=tn, fp=fp, fn=fn, tp=tp)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = confusion_counts(y_true, y_pred)
    precision = _safe_divide(c.tp, c.tp + c.fp)
    recall = _safe_divide(c.tp, c.tp + c.fn)
    specificity = _safe_divide(c.tn, c.tn + c.fp)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = _safe_divide(c.tp + c.tn, c.tp + c.tn + c.fp + c.fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": float(c.tn),
        "fp": float(c.fp),
        "fn": float(c.fn),
        "tp": float(c.tp),
    }


def expected_incident_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 20.0,
) -> float:
    c = confusion_counts(y_true, y_pred)
    return float(c.fp * false_positive_cost + c.fn * false_negative_cost)


def threshold_sweep(
    y_true: np.ndarray,
    probability: np.ndarray,
    thresholds: np.ndarray | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 20.0,
) -> list[dict[str, float]]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        row = binary_metrics(y_true, prediction)
        row["threshold"] = float(threshold)
        row["cost"] = expected_incident_cost(
            y_true,
            prediction,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        rows.append(row)
    return rows
