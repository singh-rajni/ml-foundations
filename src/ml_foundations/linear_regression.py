"""Vectorized linear regression trained with batch gradient descent."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 1-D or 2-D array")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


def _as_1d_float(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    if not np.isfinite(y).all():
        raise ValueError("y must contain only finite values")
    return y


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean((y_pred - y_true) ** 2))


@dataclass
class LinearRegressionGD:
    """Ordinary least squares optimized with batch gradient descent.

    Parameters
    ----------
    learning_rate:
        Step size used in each gradient update.
    max_iter:
        Maximum number of parameter updates.
    tol:
        Stop when the absolute loss change is at most this value.
    fit_intercept:
        Add a column of ones when True.
    """

    learning_rate: float = 0.05
    max_iter: int = 5000
    tol: float = 1e-10
    fit_intercept: bool = True
    coef_: np.ndarray | None = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d_float(X)
        if self.fit_intercept:
            return np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X_design = self._design(X)
        y = _as_1d_float(y)
        if len(X_design) != len(y):
            raise ValueError("X and y must contain the same number of rows")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")

        theta = np.zeros(X_design.shape[1], dtype=float)
        self.loss_history_ = []
        previous_loss = np.inf

        for _ in range(self.max_iter):
            prediction = X_design @ theta
            error = prediction - y
            gradient = (2.0 / len(y)) * (X_design.T @ error)
            theta -= self.learning_rate * gradient
            loss = float(np.mean((X_design @ theta - y) ** 2))
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        X = _as_2d_float(X)
        return X @ self.coef_ + self.intercept_
