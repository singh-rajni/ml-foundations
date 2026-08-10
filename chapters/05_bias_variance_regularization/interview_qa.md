# Bias, Variance, and Regularization Interview Questions and Model Answers

## 1. What is bias in the bias-variance framework?

Bias is systematic error caused by restrictive assumptions or inadequate features. A high-bias model performs poorly even on training data.

## 2. What is variance?

Variance is sensitivity to the particular training sample. A high-variance model can fit training data very well but generalize poorly.

## 3. How do learning curves identify high bias?

Training and validation performance converge at a poor level as more data is added. A more expressive model or better features may be required.

## 4. How do learning curves identify high variance?

Training performance remains strong while validation performance is weaker, with a meaningful gap. More representative data or stronger regularization may help.

## 5. Why is a train-validation gap not proof of overfitting?

Leakage, duplicates, group overlap, temporal shift, inconsistent preprocessing, and label-policy changes can also create a gap. Investigate the evaluation design first.

## 6. What does L2 regularization do?

It adds a squared-coefficient penalty, shrinking weights and improving stability. It discourages extreme reliance on individual features.

## 7. What does L1 regularization do?

It adds an absolute-coefficient penalty and can set coefficients exactly to zero, producing sparse solutions.

## 8. Why not regularize the intercept?

The intercept represents the baseline score. Penalizing it can force an arbitrary baseline shift unrelated to feature complexity. Most implementations exclude it.

## 9. Why scale before regularization?

Without scaling, the same penalty affects features differently because coefficient magnitude depends on feature units.

## 10. What is Elastic Net?

A combination of L1 and L2 penalties. It can create sparsity while handling groups of correlated features more smoothly than pure L1.

## 11. How do you choose regularization strength?

Use validation or cross-validation with the correct temporal or group structure, select a metric aligned with the decision, then confirm stability and segment performance.

## 12. What happens when regularization is too strong?

Coefficients shrink excessively, probabilities move toward the baseline, and the model underfits.

## 13. How does data size affect variance?

More representative data often reduces variance because the model is less dependent on individual observations. It does not automatically fix high bias or bad labels.

## 14. Why can L1 feature selection be unstable?

When features are correlated, small sample changes may cause L1 to choose one feature over another even when both contain similar information.

## 15. How would you decide whether to approve a more complex model?

Compare business value, uncertainty, segment behavior, calibration, latency, cost, explainability, reliability, skill requirements, and rollback risk. Approve complexity only when the incremental value is material and operationally supportable.
