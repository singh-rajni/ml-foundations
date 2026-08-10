# Linear Regression Exercises

## Calculation

Given $x=[1,2,3]$, $y=[2,4,5]$, $w=0$, $b=0$, and learning rate $0.05$:

1. Calculate all predictions.
2. Calculate MSE.
3. Calculate $dJ/dw$ and $dJ/db$.
4. Perform one parameter update.
5. Calculate the new MSE.

## Coding

1. Reimplement `predict`, `mse`, and `fit` without opening the reference script.
2. Add mini-batch gradient descent.
3. Add early stopping using validation MSE.
4. Add a `score` method that returns R-squared.
5. Create a test showing the learned coefficients are close to `np.linalg.lstsq`.

## Analysis

1. Add one extreme outlier. Compare MSE and MAE before and after.
2. Create two highly correlated features and observe coefficient instability.
3. Train with and without feature scaling. Compare iterations to convergence.
4. Use a temporal split and compare it with a random split.

## Director case

A complex model reduces RMSE from 4.0 hours to 3.8 hours, but triples serving cost, removes coefficient-level explanations, and requires a separate GPU service. Prepare a two-minute recommendation covering value, risk, adoption, reliability, and experiment design.
