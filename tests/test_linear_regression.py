import numpy as np

from ml_foundations.linear_regression import LinearRegressionGD, mean_squared_error


def test_linear_regression_recovers_simple_line():
    x = np.linspace(-2, 2, 100).reshape(-1, 1)
    y = 1.5 + 2.25 * x[:, 0]
    model = LinearRegressionGD(learning_rate=0.08, max_iter=5000, tol=1e-12).fit(x, y)
    assert np.isclose(model.intercept_, 1.5, atol=1e-5)
    assert np.allclose(model.coef_, [2.25], atol=1e-5)
    assert mean_squared_error(y, model.predict(x)) < 1e-9


def test_linear_regression_requires_fit_before_predict():
    model = LinearRegressionGD()
    try:
        model.predict(np.array([[1.0]]))
    except RuntimeError:
        return
    raise AssertionError("predict should fail before fit")
