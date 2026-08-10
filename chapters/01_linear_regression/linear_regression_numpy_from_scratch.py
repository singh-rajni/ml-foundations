"""Linear regression with MSE and batch gradient descent using NumPy only.

Run:
    python linear_regression_numpy_from_scratch.py
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit before transform")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.01, max_steps: int = 5000,
                 tolerance: float = 1e-10, patience: int = 30):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.patience = patience
        self.theta_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    @staticmethod
    def _validate(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values")
        return X, y

    @staticmethod
    def add_intercept(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X, y = self._validate(X, y)
        Xd = self.add_intercept(X)
        theta = np.zeros(Xd.shape[1], dtype=float)
        self.loss_history_ = []
        best = np.inf
        stale = 0

        for _ in range(self.max_steps):
            pred = Xd @ theta
            error = pred - y
            loss = float(np.mean(error ** 2))
            gradient = (2.0 / len(y)) * (Xd.T @ error)
            theta -= self.learning_rate * gradient
            self.loss_history_.append(loss)

            if best - loss > self.tolerance:
                best = loss
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.theta_ = theta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=float)
        return self.add_intercept(X) @ self.theta_

    @property
    def intercept_(self) -> float:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted")
        return float(self.theta_[0])

    @property
    def coef_(self) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted")
        return self.theta_[1:].copy()


def mse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(y)) ** 2))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y, pred)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def finite_difference_gradient(Xd: np.ndarray, y: np.ndarray, theta: np.ndarray,
                               epsilon: float = 1e-6) -> np.ndarray:
    numerical = np.zeros_like(theta)
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = epsilon
        loss_plus = mse(y, Xd @ (theta + step))
        loss_minus = mse(y, Xd @ (theta - step))
        numerical[j] = (loss_plus - loss_minus) / (2 * epsilon)
    return numerical


def main() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(0.5, 10.0, 60).reshape(-1, 1)
    y = 2.75 + 2.35 * x[:, 0] + rng.normal(0, 1.55, size=len(x))

    model = LinearRegressionGD(learning_rate=0.012, max_steps=5000,
                               tolerance=1e-12, patience=100)
    model.fit(x, y)
    pred = model.predict(x)

    print(f"intercept: {model.intercept_:.4f}")
    print(f"coefficient: {model.coef_[0]:.4f}")
    print(f"MSE:  {mse(y, pred):.4f}")
    print(f"RMSE: {rmse(y, pred):.4f}")
    print(f"MAE:  {mae(y, pred):.4f}")
    print(f"R2:   {r2(y, pred):.4f}")
    print(f"iterations: {len(model.loss_history_)}")

    Xd = model.add_intercept(x)
    closed = np.linalg.pinv(Xd) @ y
    print("pseudoinverse coefficients:", closed)

    theta_test = np.array([0.3, -0.2])
    analytical = (2.0 / len(y)) * Xd.T @ (Xd @ theta_test - y)
    numerical = finite_difference_gradient(Xd, y, theta_test)
    print("max gradient-check error:", np.max(np.abs(analytical - numerical)))


if __name__ == "__main__":
    main()
