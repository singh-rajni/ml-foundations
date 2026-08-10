# Threshold and Calibration Interview Questions and Model Answers

## 1. Why is 0.5 not always the correct threshold?

The best threshold depends on class prevalence, error costs, capacity, safety, and product policy. The model estimates a score or probability; the organization chooses the action rule.

## 2. Where should the threshold be tuned?

On validation data or through cross-validation, never on the final test set. For time-dependent systems, use a temporal validation period.

## 3. What is calibration?

Calibration is agreement between predicted probabilities and observed event frequencies. Among cases predicted near 0.7, about 70 percent should be positive over time if the model is calibrated for that population.

## 4. Can a model have high AUC and poor calibration?

Yes. A monotonic transformation can preserve ranking and therefore AUC while changing probability values substantially.

## 5. What is a reliability diagram?

It bins predictions and plots observed positive rate against mean predicted probability. Counts per bin should also be inspected because sparse bins are noisy.

## 6. What is Brier score?

It is mean squared error between predicted probability and binary outcome. It rewards accurate probabilistic predictions but mixes discrimination and calibration.

## 7. What is expected calibration error?

It is a weighted average of absolute gaps between mean confidence and observed rate across bins. It is simple but depends on binning and may hide local failures.

## 8. Sigmoid versus isotonic calibration?

Sigmoid calibration is smoother and often better with less data. Isotonic is more flexible and can capture non-sigmoid patterns but needs more calibration data and can overfit.

## 9. Why must calibration use independent data?

A model is usually more confident and accurate on its training data. Calibrating on those same predictions would produce an optimistically biased mapping.

## 10. How do you choose a threshold with uncertain costs?

Perform sensitivity analysis across plausible false-positive and false-negative costs, add hard safety and capacity constraints, and prefer policies stable across a reasonable cost range.

## 11. What if review capacity changes each day?

Use ranking and a capacity-aware top-k policy or dynamic threshold, but govern the policy carefully and monitor recall, queue effects, and calibration under the changing selection mechanism.

## 12. Can calibration degrade after deployment?

Yes. Prevalence shifts, data drift, policy changes, label changes, and model updates can all alter calibration. Monitor and recalibrate using recent representative data.

## 13. Should each business unit have a separate threshold?

Only when base rates, costs, or workflows genuinely differ and the segmentation is statistically supported, lawful, maintainable, and governed. Otherwise it can create instability and hidden inequities.

## 14. What is a reject option?

It is a policy that sends uncertain cases to human review rather than forcing every probability into an automated yes/no decision.

## 15. How would you roll out a new threshold?

Evaluate offline, run shadow comparisons, estimate capacity impact, canary to a limited group, monitor errors and queue health, maintain an audit trail, and preserve rapid rollback.
