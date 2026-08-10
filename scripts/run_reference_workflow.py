"""Run the capstone reference workflow and print a compact scorecard."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ml_foundations.calibration import expected_calibration_error
from ml_foundations.data import FEATURE_COLUMNS, temporal_split
from ml_foundations.metrics import binary_metrics, threshold_sweep


def main() -> None:
    data = pd.read_csv(REPO / "data/incident_priority_synthetic.csv", parse_dates=["timestamp"])
    train, validation, test = temporal_split(data)

    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=0.5, max_iter=3000),
    )
    model = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=5)
    model.fit(train[FEATURE_COLUMNS], train["is_p1"])

    p_validation = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    sweep = pd.DataFrame(
        threshold_sweep(
            validation["is_p1"].to_numpy(),
            p_validation,
            false_positive_cost=1.0,
            false_negative_cost=25.0,
        )
    )
    feasible = sweep[sweep["recall"] >= 0.85]
    selected = (feasible if len(feasible) else sweep).sort_values(["cost", "threshold"]).iloc[0]
    threshold = float(selected["threshold"])

    y_test = test["is_p1"].to_numpy()
    p_test = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    prediction = (p_test >= threshold).astype(int)
    threshold_metrics = binary_metrics(y_test, prediction)

    scorecard = {
        "threshold": threshold,
        "test_prevalence": float(np.mean(y_test)),
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "average_precision": float(average_precision_score(y_test, p_test)),
        "log_loss": float(log_loss(y_test, p_test)),
        "brier_score": float(brier_score_loss(y_test, p_test)),
        "expected_calibration_error": float(expected_calibration_error(y_test, p_test)),
        **threshold_metrics,
        "alerts": int(prediction.sum()),
    }
    print(json.dumps(scorecard, indent=2))


if __name__ == "__main__":
    main()
