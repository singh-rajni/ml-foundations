# Bias, Variance, and Regularization Exercises

## Coding

1. Generate a noisy curved dataset.
2. Fit polynomial models of degree 1, 3, 8, and 15.
3. Plot training and validation error by degree.
4. Add Ridge regularization and compare.
5. Build learning curves using increasing training sizes.
6. Fit L1 and L2 logistic regression to the incident data.
7. Plot coefficient magnitude against `C`.

## Analysis

1. Identify the degree with minimum validation error.
2. Explain why training error cannot select model complexity.
3. Compare feature coefficients before and after scaling.
4. Add a leaked post-triage feature and observe the unrealistic improvement.
5. Remove the leaked feature and document the lesson.

## Director case

A team proposes a highly complex ensemble because training performance is nearly perfect. Validation performance is only marginally better than regularized logistic regression, and the ensemble is difficult to calibrate. Prepare a recommendation and an experiment that tests whether the added complexity creates measurable operational value.
