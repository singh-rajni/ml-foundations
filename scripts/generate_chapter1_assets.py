from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ml_foundations.linear_regression import LinearRegressionGD


def main() -> None:
    assets = REPO / "chapters/01_linear_regression/assets"
    assets.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    x = np.linspace(0.5, 10.0, 70)
    y = 2.75 + 2.35 * x + rng.normal(0, 1.55, size=x.shape)
    X = x.reshape(-1, 1)

    model = LinearRegressionGD(learning_rate=0.012, max_iter=3000, tol=1e-12).fit(X, y)
    prediction = model.predict(X)

    plt.figure(figsize=(8, 4.8))
    plt.scatter(x, y, s=34, label="Observed")
    plt.plot(x, prediction, linewidth=2.5, label="Fitted line")
    plt.xlabel("Impacted services")
    plt.ylabel("Resolution time in hours")
    plt.title("Linear regression fit")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets / "regression_fit.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(model.loss_history_)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("MSE on log scale")
    plt.title("Gradient descent convergence")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(assets / "gradient_descent_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for learning_rate in [0.001, 0.012, 0.06]:
        trial = LinearRegressionGD(learning_rate=learning_rate, max_iter=300, tol=0).fit(X, y)
        history = np.minimum(np.asarray(trial.loss_history_), 1e8)
        plt.plot(history, label=f"learning_rate={learning_rate}")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("MSE on log scale")
    plt.title("Learning-rate comparison")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(assets / "learning_rate_comparison.png", dpi=160)
    plt.close()

    residual = y - prediction
    plt.figure(figsize=(8, 4.5))
    plt.scatter(prediction, residual, s=34)
    plt.axhline(0.0, linewidth=2)
    plt.xlabel("Predicted resolution time")
    plt.ylabel("Residual: actual - predicted")
    plt.title("Residual diagnostic")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(assets / "residual_plot.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
