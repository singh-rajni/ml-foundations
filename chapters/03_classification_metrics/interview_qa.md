# Classification Metrics Interview Questions and Model Answers

## 1. Why can accuracy be misleading?

When positives are rare, predicting the majority class can produce high accuracy while detecting no positives. Accuracy also treats false positives and false negatives as equally costly.

## 2. Precision versus recall?

Precision asks whether positive predictions are trustworthy. Recall asks whether actual positives are found. The correct emphasis depends on the cost of unnecessary action versus missed action.

## 3. Why is F1 a harmonic mean?

The harmonic mean becomes low when either precision or recall is low, so a model cannot obtain a high F1 by excelling at only one. It still assumes equal importance and ignores true negatives.

## 4. When would you use F-beta?

When the business explicitly values recall and precision differently. A beta greater than one emphasizes recall; a beta below one emphasizes precision. The choice should be tied to cost or capacity.

## 5. What does ROC-AUC measure?

It summarizes ranking discrimination across thresholds. A common interpretation is the probability that a randomly selected positive is scored above a randomly selected negative.

## 6. Why can ROC-AUC be optimistic for rare events?

The false positive rate divides by a very large number of negatives, so a tolerable-looking rate can still produce too many false alerts in absolute terms.

## 7. Why is a precision-recall curve useful for imbalanced data?

It directly displays the trade-off between finding positives and maintaining trust in positive predictions. Its baseline reflects positive prevalence.

## 8. Is PR-AUC always better than ROC-AUC?

No. They answer different questions. PR-AUC emphasizes positive retrieval; ROC-AUC gives a broader ranking view. Report the metric aligned with the decision and usually include both.

## 9. Does AUC choose a threshold?

No. AUC evaluates ranking over thresholds. A deployment threshold must be selected using validation data and operational cost, capacity, safety, or policy constraints.

## 10. What is specificity?

Specificity is the fraction of actual negatives correctly predicted negative: $TN/(TN+FP)$. It is useful when false-positive control matters.

## 11. What is balanced accuracy?

It is the average of sensitivity and specificity. It gives equal weight to both classes, but still may not match business cost.

## 12. What is the baseline for average precision?

A non-informative ranking has expected precision near the positive prevalence. Therefore average precision must be interpreted relative to prevalence and the evaluation sample.

## 13. How do you compare metrics across time if prevalence changes?

Report prevalence, threshold-specific counts, calibration, segment metrics, and cost. Avoid attributing every metric change to model drift when the population changed.

## 14. How do you quantify uncertainty in a metric?

Use confidence intervals, often with appropriate bootstrap resampling. For time-dependent data, preserve temporal or cluster structure rather than resampling observations blindly.

## 15. What would you put on an executive dashboard?

Business outcomes, missed critical incidents, alert volume, precision and recall at the active threshold, calibration, key segment disparities, latency, data freshness, incident trends, and rollback status.
