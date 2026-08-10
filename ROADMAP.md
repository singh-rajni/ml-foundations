# Seven-Day Study Roadmap

## Daily time budget

| Block | Time | Activity |
|---|---:|---|
| Concept | 45 minutes | Read intuition and mathematical explanation |
| Derivation | 30 minutes | Reproduce equations by hand |
| Implementation | 75 minutes | Run and rewrite code |
| Interview | 30 minutes | Answer questions aloud |
| Reflection | 15 minutes | Record errors, decisions, and next action |

## Day 1 - Linear regression

Outcome: predict a continuous quantity and optimize MSE using gradient descent.

Pass criteria:

- Derive gradients for slope and intercept.
- Implement vectorized batch gradient descent.
- Explain learning-rate failure modes.
- Compare gradient descent with a closed-form solution.

## Day 2 - Logistic regression

Outcome: predict a probability for a binary outcome.

Pass criteria:

- Explain logits, odds, sigmoid, and cross-entropy.
- Implement logistic regression with NumPy.
- Compare probabilities and coefficients with scikit-learn.
- Explain the role of feature scaling and regularization.

## Day 3 - Classification metrics

Outcome: select metrics that reflect the cost of incident decisions.

Pass criteria:

- Calculate precision, recall, specificity, and F1 from a confusion matrix.
- Explain ROC-AUC and PR-AUC without saying one is always better.
- Select a primary metric for rare P1 incidents.
- Report segment-level and operational metrics.

## Day 4 - Thresholds and calibration

Outcome: convert probabilities into actions responsibly.

Pass criteria:

- Select a threshold using explicit false-positive and false-negative costs.
- Create and interpret a reliability diagram.
- Explain discrimination versus calibration.
- Avoid using the test set to tune the threshold.

## Day 5 - Bias, variance, and regularization

Outcome: diagnose whether a model is too simple, too complex, or data-limited.

Pass criteria:

- Interpret training and validation learning curves.
- Explain L1 versus L2 geometrically and operationally.
- Tune regularization using validation or cross-validation.
- Identify leakage before blaming overfitting.

## Day 6 - Incident classifier capstone

Outcome: build a reproducible, cost-aware, calibrated classification workflow.

Pass criteria:

- Use a temporal train/validation/test split.
- Establish a baseline.
- Build a preprocessing and model pipeline.
- Tune threshold on validation data.
- Lock the threshold and evaluate once on test data.
- Produce a model card and rollout recommendation.

## Day 7 - Quiz and presentation

Outcome: demonstrate recall, communication, and Director-level judgment.

Pass criteria:

- Score at least 20/25 on the quiz.
- Deliver a five-minute explanation without reading.
- Answer at least five follow-up questions.
- State limitations and a safe rollout plan.
