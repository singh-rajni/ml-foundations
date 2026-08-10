"""Train binary logistic regression with NumPy and compare with scikit-learn."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ml_foundations.logistic_regression import LogisticRegressionGD


def main() -> None:
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=[-1.2, -0.7], scale=[1.0, 1.2], size=(350, 2))
    X1 = rng.normal(loc=[1.3, 1.0], scale=[1.1, 0.9], size=(350, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(350, dtype=int), np.ones(350, dtype=int)])
    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    split = 520
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    numpy_model = LogisticRegressionGD(
        learning_rate=0.12,
        max_iter=5000,
        tol=1e-10,
    ).fit(X_train, y_train)
    p_numpy = numpy_model.predict_proba(X_test)[:, 1]

    library_model = LogisticRegression(C=1e6, max_iter=5000).fit(X_train, y_train)
    p_library = library_model.predict_proba(X_test)[:, 1]

    print("NumPy coefficients:", np.round(numpy_model.coef_, 5))
    print("Library coefficients:", np.round(library_model.coef_[0], 5))
    print("NumPy log-loss:", round(log_loss(y_test, p_numpy), 6))
    print("NumPy ROC-AUC:", round(roc_auc_score(y_test, p_numpy), 6))
    print("Probability correlation:", round(np.corrcoef(p_numpy, p_library)[0, 1], 8))


if __name__ == "__main__":
    main()
