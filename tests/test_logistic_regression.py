import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ml_foundations.logistic_regression import LogisticRegressionGD, binary_cross_entropy, stable_sigmoid


def test_stable_sigmoid_extreme_values_are_finite():
    values = stable_sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.isfinite(values).all()
    assert values[0] < 1e-12
    assert np.isclose(values[1], 0.5)
    assert values[2] > 1 - 1e-12


def test_binary_cross_entropy_prefers_correct_probabilities():
    y = np.array([0, 1])
    good = binary_cross_entropy(y, np.array([0.05, 0.95]))
    bad = binary_cross_entropy(y, np.array([0.95, 0.05]))
    assert good < bad


def test_logistic_regression_learns_separable_signal():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 2))
    logit = 1.7 * X[:, 0] - 1.2 * X[:, 1] + 0.2
    probability = stable_sigmoid(logit)
    y = rng.binomial(1, probability)
    X = StandardScaler().fit_transform(X)
    model = LogisticRegressionGD(learning_rate=0.15, max_iter=5000, tol=1e-10).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert roc_auc_score(y, p) > 0.80
    assert model.loss_history_[-1] < model.loss_history_[0]
