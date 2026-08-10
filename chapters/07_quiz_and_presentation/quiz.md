# 25-Question Mastery Quiz

Do not use notes or code for the first attempt. Show calculation steps where requested.

## Linear regression

### 1. Concept

State the simple linear regression equation and define every symbol.

### 2. Calculation

For actual values `[2, 4, 7]` and predictions `[3, 5, 5]`, calculate residuals using `prediction - actual`, MSE, RMSE, and MAE.

### 3. Derivation

Starting from $J(w,b)=\frac{1}{n}\sum(wx_i+b-y_i)^2$, write $\partial J/\partial w$ and $\partial J/\partial b$.

### 4. Debugging

Training loss becomes `NaN` after five iterations. Give four plausible causes and the first diagnostic you would run for each.

### 5. Judgment

A tree ensemble improves RMSE by 2 percent over linear regression but costs five times more to serve. What information is required before approving it?

## Logistic regression

### 6. Concept

Explain probability, odds, log-odds, and logit for logistic regression.

### 7. Calculation

Calculate sigmoid for $z=0$ and state why this value matters for the default decision boundary.

### 8. Loss

Write binary cross-entropy for one observation and explain how it treats a confident wrong prediction.

### 9. Gradient

Write the vectorized gradient for unregularized binary logistic regression.

### 10. Comparison

Your NumPy coefficients differ from scikit-learn. List five configuration or implementation differences to check.

## Classification metrics

### 11. Calculation

Given TP=18, FP=42, FN=2, TN=938, calculate accuracy, precision, recall, specificity, and F1.

### 12. Concept

Why can a 98 percent accurate model be useless for P1 detection?

### 13. Curves

Explain the axes of ROC and precision-recall curves.

### 14. Judgment

Why might average precision be a better primary discrimination metric than ROC-AUC for a 2 percent positive rate?

### 15. Operations

At the active threshold, precision is 10 percent and recall is 95 percent. What operational facts determine whether this is acceptable?

## Thresholds and calibration

### 16. Threshold

Why is 0.5 not a universal threshold?

### 17. Protocol

Which dataset should be used to select the threshold, and what must happen before evaluating on the test set?

### 18. Calibration

A model assigns probability 0.8 to 100 similar incidents, but only 50 become P1. What does this suggest?

### 19. Metrics

State what Brier score measures and one limitation.

## Bias, variance, and regularization

### 20. Diagnosis

Training and validation errors are both high and close. What is the likely diagnosis and what interventions should be tested?

### 21. Diagnosis

Training error is very low but validation error is much higher. Give four explanations other than simply saying overfitting.

### 22. Regularization

Compare L1 and L2 regularization, including coefficient behavior and one operational implication.

## Capstone and leadership

### 23. Leakage

Name five incident fields that could leak future information into a model trained to score at incident creation.

### 24. Rollout

Design a four-stage rollout from offline evaluation to controlled production use.

### 25. Executive response

The model misses a high-impact P1 incident during pilot. Give a two-minute response covering immediate action, diagnosis, communication, and prevention.
