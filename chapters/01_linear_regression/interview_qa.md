# Linear Regression Interview Questions and Model Answers

## 1. What does the slope mean?

For a one-unit increase in the feature, the model changes its prediction by the slope, holding other features constant in multiple regression. The interpretation is associational unless the design supports causal inference.

## 2. Why minimize squared error?

Squared error is smooth, differentiable, and gives a closed-form solution under ordinary least squares. It also corresponds to maximum likelihood when residuals are independent Gaussian noise with constant variance. Its weakness is sensitivity to large errors and outliers.

## 3. Why is there a factor of 2 in the gradient?

Differentiating a squared residual produces the factor of 2. Some texts define the loss with $1/(2n)$ so that the 2 cancels. Both definitions reach the same optimum if the learning rate is adjusted consistently.

## 4. What does the intercept represent?

It is the prediction when every feature is zero. The numeric value may not be meaningful when zero is outside the observed domain or when features are standardized.

## 5. What assumptions matter?

Important assumptions for inference include linearity, independent errors, constant error variance, limited multicollinearity, and approximately normal residuals for certain confidence intervals. Prediction can still be useful when some inference assumptions fail, but diagnostics and uncertainty must be handled carefully.

## 6. What is multicollinearity?

Strong correlation among features makes individual coefficient estimates unstable. Predictions may remain adequate, but interpretation becomes unreliable. Remedies include feature removal, combining features, collecting more data, or regularization.

## 7. Why standardize features for gradient descent?

Features on similar scales create a better-conditioned optimization landscape. Gradient descent can move efficiently rather than zigzagging across dimensions with very different magnitudes.

## 8. Can R-squared be negative?

Yes, on evaluation data it can be negative when the model is worse than predicting the target mean. A high training R-squared does not prove generalization.

## 9. MSE or MAE?

Choose based on decision cost. MSE emphasizes large errors and is smooth. MAE is more robust and directly represents average absolute error. Report both when the operational impact is asymmetric or outlier-sensitive.

## 10. Why not invert $X^TX$ directly?

Explicit inversion is numerically less stable and unnecessary. Use `lstsq`, QR, SVD, or a pseudoinverse routine.

## 11. How do you detect nonlinear structure?

Plot residuals against predictions and important features. Systematic curves, segment patterns, or remaining structure suggest transformations, interactions, splines, or a nonlinear model.

## 12. What is heteroscedasticity?

Residual variance changes with the prediction or feature values. It affects uncertainty estimates and may indicate transformations, weighted regression, segment-specific models, or robust standard errors.

## 13. When would you prefer a linear model over gradient boosting?

When interpretability, stability, latency, governance, small data, or operational simplicity dominate and the complex model does not materially change a business decision.

## 14. How do you prove the gradient implementation is correct?

Use finite-difference gradient checking on a small problem, compare convergence with a closed-form result, add unit tests, and inspect whether loss decreases under a reasonable learning rate.

## 15. How would you monitor a deployed regression model?

Monitor input distributions, missingness, prediction distribution, residual metrics after labels arrive, segment-level error, calibration of prediction intervals if used, latency, failures, and whether users act on the predictions.
