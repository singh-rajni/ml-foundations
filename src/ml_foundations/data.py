"""Synthetic incident data used throughout the repository."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "service_criticality",
    "customer_impact_pct",
    "affected_services",
    "error_rate_pct",
    "latency_increase_pct",
    "recent_deploy",
    "repeat_incident_30d",
    "monitoring_confidence",
]


def make_incident_priority_data(n_samples: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible, imbalanced binary incident dataset.

    The target `is_p1` is generated from a logistic probability. A time trend
    introduces modest distribution shift so that temporal validation is useful.
    """
    if n_samples < 300:
        raise ValueError("n_samples must be at least 300")
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2024-01-01", periods=n_samples, freq="3h")
    time_fraction = np.linspace(0.0, 1.0, n_samples)

    service_criticality = rng.integers(1, 6, n_samples)
    customer_impact_pct = np.clip(rng.beta(1.2, 6.0, n_samples) * 100.0 + 5.0 * time_fraction, 0, 100)
    affected_services = np.clip(rng.poisson(2.5 + 1.5 * time_fraction, n_samples) + 1, 1, 25)
    error_rate_pct = np.clip(rng.gamma(1.7, 3.0, n_samples) + 1.5 * time_fraction, 0, 100)
    latency_increase_pct = np.clip(rng.gamma(2.0, 12.0, n_samples), 0, 250)
    recent_deploy = rng.binomial(1, 0.26 + 0.07 * time_fraction, n_samples)
    repeat_incident_30d = rng.binomial(1, 0.19, n_samples)
    monitoring_confidence = np.clip(rng.beta(7.0, 2.2, n_samples), 0.05, 0.999)

    logit = (
        -8.2
        + 0.58 * service_criticality
        + 0.030 * customer_impact_pct
        + 0.115 * affected_services
        + 0.075 * error_rate_pct
        + 0.0065 * latency_increase_pct
        + 0.78 * recent_deploy
        + 0.60 * repeat_incident_30d
        + 0.85 * (service_criticality >= 5) * (customer_impact_pct >= 25)
        - 0.65 * monitoring_confidence
        + 0.55 * time_fraction
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    is_p1 = rng.binomial(1, probability, n_samples)

    data = pd.DataFrame(
        {
            "incident_id": [f"INC-{i:06d}" for i in range(1, n_samples + 1)],
            "timestamp": timestamp,
            "service_criticality": service_criticality,
            "customer_impact_pct": np.round(customer_impact_pct, 3),
            "affected_services": affected_services,
            "error_rate_pct": np.round(error_rate_pct, 3),
            "latency_increase_pct": np.round(latency_increase_pct, 3),
            "recent_deploy": recent_deploy,
            "repeat_incident_30d": repeat_incident_30d,
            "monitoring_confidence": np.round(monitoring_confidence, 5),
            "is_p1": is_p1,
        }
    )
    return data


def temporal_split(
    data: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")
    ordered = data.sort_values("timestamp").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:validation_end].copy(),
        ordered.iloc[validation_end:].copy(),
    )
