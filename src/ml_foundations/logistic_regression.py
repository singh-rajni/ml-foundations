"""Binary logistic regression implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    if len(y_true) != len(probability):
        raise ValueError("y_true and probability must have the same length")
    if not set(np.unique(y_true)).issubset({0.0, 1.0}):
        raise ValueError("y_true must contain only 0 and 1")
    eps = np.finfo(float).eps
    probability = np.clip(probability, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(probability) + (1.0 - y_true) * np.log(1.0 - probability)))


@dataclass
class LogisticRegressionGD:
    """Binary logistic regression with L1 or L2 regularization.

    The intercept is not regularized. L1 uses a subgradient at zero.
    """

    learning_rate: float = 0.1
    max_iter: int = 5000
    tol: float = 1e-9
    l1: float = 0.0
    l2: float = 0.0
    fit_intercept: bool = True
    coef_: np.ndarray | None = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)

    @staticmethod
    def _prepare_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1-D or 2-D array")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_X(X)
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X_design = self._design(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X_design) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("y must contain only 0 and 1")
        if self.learning_rate <= 0 or self.max_iter <= 0:
            raise ValueError("learning_rate and max_iter must be positive")
        if self.l1 < 0 or self.l2 < 0:
            raise ValueError("regularization strengths cannot be negative")

        theta = np.zeros(X_design.shape[1], dtype=float)
        self.loss_history_ = []
        previous_loss = np.inf

        for _ in range(self.max_iter):
            probability = stable_sigmoid(X_design @ theta)
            gradient = (X_design.T @ (probability - y)) / len(y)

            penalty_slice = slice(1, None) if self.fit_intercept else slice(None)
            regularized = theta[penalty_slice]
            gradient[penalty_slice] += self.l2 * regularized
            gradient[penalty_slice] += self.l1 * np.sign(regularized)

            theta -= self.learning_rate * gradient
            probability = stable_sigmoid(X_design @ theta)
            loss = binary_cross_entropy(y, probability)
            loss += 0.5 * self.l2 * float(np.sum(theta[penalty_slice] ** 2))
            loss += self.l1 * float(np.sum(np.abs(theta[penalty_slice])))
            self.loss_history_.append(loss)
            if abs(previous_loss - loss) <= self.tol:
                break
            previous_loss = loss

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = theta.copy()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit must be called before prediction")
        X = self._prepare_X(X)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = stable_sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
