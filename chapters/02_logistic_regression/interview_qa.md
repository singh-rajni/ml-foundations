# Logistic Regression Interview Questions and Model Answers

## 1. Why is logistic regression called regression if it is used for classification?

It models a continuous quantity, the log-odds of the positive class, as a linear function of features. A threshold then turns the probability into a class decision.

## 2. What is a logit?

The logit is the logarithm of odds: $\log(p/(1-p))$. Logistic regression assumes this value is linear in the features.

## 3. Why use sigmoid?

Sigmoid maps any real score into a value between 0 and 1 and has a derivative $\sigma(z)(1-\sigma(z))$, which supports efficient gradient-based optimization.

## 4. Why use cross-entropy?

It is the negative log-likelihood for Bernoulli outcomes. It penalizes confident wrong predictions strongly and yields the gradient $X^T(p-y)/n$.

## 5. Why can sigmoid overflow?

For large negative values, `exp(-z)` can be enormous. A branch-based stable implementation avoids evaluating the unstable expression.

## 6. What does a positive coefficient mean?

Holding other features constant, an increase in the feature increases the log-odds and therefore the probability of the positive class. The exact probability change depends on the current score.

## 7. How do you convert a coefficient to an odds ratio?

Use $\exp(w_j)$. For example, an odds ratio of 1.5 means odds multiply by 1.5 for a one-unit feature increase, holding other variables constant.

## 8. Is logistic regression a linear classifier?

Yes, its decision boundary is linear in the original feature space unless nonlinear transformations or interaction features are added.

## 9. What is perfect separation?

A feature combination separates classes without error, so unregularized maximum-likelihood coefficients can grow without bound. Regularization or a different estimation strategy is needed.

## 10. Why is scaling important?

It improves optimizer conditioning, makes regularization act more comparably across features, and supports meaningful coefficient comparisons.

## 11. What does `C` mean in scikit-learn logistic regression?

`C` is inverse regularization strength. Smaller `C` means stronger regularization. Always verify solver and penalty compatibility.

## 12. How do class weights work?

They multiply the contribution of selected observations or classes to the loss. They can improve attention to rare cases, but they change the fitted score distribution and do not remove the need for threshold tuning or calibration checks.

## 13. How do you handle nonlinear relationships?

Add transformations, splines, interactions, or switch to a nonlinear model. Validate whether added complexity materially improves the decision.

## 14. Does a predicted probability of 0.8 mean 80 percent certainty for this one case?

It is a model estimate, not a guarantee. Calibration means that among comparable cases assigned about 0.8, roughly 80 percent should be positive over time. Epistemic and data uncertainty still remain.

## 15. How would you debug poor convergence?

Check scaling, learning rate, regularization, label values, gradient signs, finite inputs, class separation, and whether loss decreases on a small test problem. Compare analytical and numerical gradients.
