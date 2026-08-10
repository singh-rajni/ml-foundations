# Logistic Regression Exercises

## Calculation

For score $z=1.2$:

1. Calculate the sigmoid probability.
2. Calculate cross-entropy for $y=1$.
3. Calculate cross-entropy for $y=0$.
4. Explain which outcome is more surprising to the model.

## Coding

1. Write a stable sigmoid without copying the reference.
2. Implement binary cross-entropy.
3. Implement the vectorized gradient.
4. Add L2 regularization while excluding the intercept.
5. Add `predict_proba` and a configurable threshold.
6. Compare coefficients and probabilities with scikit-learn using matching settings.

## Analysis

1. Train with raw and standardized features. Compare convergence.
2. Vary `C` in scikit-learn and inspect coefficient magnitude.
3. Create an interaction feature and assess whether it improves validation log-loss.
4. Change the class balance and compare accuracy with recall and average precision.

## Director case

A logistic regression model is highly interpretable and calibrated but has slightly lower PR-AUC than a tree ensemble. Prepare a recommendation that covers the decision cost, explanation requirements, calibration, maintenance, rollout, and whether the difference is statistically and operationally meaningful.
