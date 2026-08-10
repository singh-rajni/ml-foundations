# Five-Minute Presentation Guide

## Presentation title

From Linear Models to a Governed Incident-Priority Classifier

## Slide 1 - Problem and outcome, 40 seconds

- P1 incidents are rare and high-impact.
- Goal: prioritize faster human review, not replace incident commanders.
- Success: fewer missed P1s with manageable alert burden.

## Slide 2 - Foundations, 50 seconds

- Linear regression introduced prediction, residuals, MSE, and gradient descent.
- Logistic regression converted a linear score to probability using sigmoid.
- Cross-entropy trained probability estimates for binary outcomes.

## Slide 3 - Evaluation, 60 seconds

- Accuracy is inadequate for rare incidents.
- Use average precision and ROC-AUC for ranking.
- Use recall, precision, and alert count at the operating threshold.
- Use Brier score and reliability diagrams for probability quality.

## Slide 4 - Decision policy, 55 seconds

- Probability is not an action.
- Select threshold on validation data using false-negative and false-positive costs plus capacity and recall guardrails.
- Freeze policy before test evaluation.

## Slide 5 - Generalization, 45 seconds

- Diagnose high bias and variance with learning curves.
- Use L1/L2 regularization and leakage-safe pipelines.
- Prefer the simplest model that creates material decision value.

## Slide 6 - Production design and recommendation, 50 seconds

- Validate data, score, calibrate, apply a versioned threshold, and present to human reviewers.
- Log model, features, score, decision, override, and final label.
- Recommend shadow mode, then a human-in-the-loop pilot with fallback and rollback.

## Closing, 20 seconds

The key learning is that model development is only one part of the solution. The production artifact is a governed decision system connecting data quality, probability, policy, people, and measurable outcomes.

## Likely follow-up questions

1. Why logistic regression rather than gradient boosting?
2. What would change the threshold?
3. How would you monitor calibration?
4. What is the highest leakage risk?
5. When would you approve automation?
