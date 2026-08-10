# Capstone Interview Questions and Model Answers

## 1. Why start with logistic regression?

It provides a strong, fast, explainable probability baseline, is easy to operate, and exposes data and evaluation problems before adding model complexity.

## 2. Why use a temporal split?

Deployment predicts future incidents. A temporal split better simulates future data and reveals drift or policy changes that a random split can hide.

## 3. Why not oversample before the split?

That can copy or synthesize information across evaluation boundaries and cause leakage. Split first, then apply any resampling only inside training folds.

## 4. Why use a pipeline?

It keeps preprocessing and model fitting together, reducing leakage and ensuring the same transformations are applied during training, validation, and serving.

## 5. Why not optimize accuracy?

P1 incidents are rare. A majority-class model can be highly accurate while missing every P1. The decision requires recall, precision, alert volume, cost, and calibration.

## 6. Why calibrate probabilities?

Threshold cost, staffing, and risk tiers may depend on absolute probability. Good ranking alone does not guarantee reliable probabilities.

## 7. How is the threshold chosen?

On validation data using an explicit cost function and recall or capacity guardrails. It is frozen before final test evaluation.

## 8. What is the largest leakage risk?

Using fields created after triage or escalation, such as final priority, resolver outcome, or resolution duration. Timestamp and lineage review are mandatory.

## 9. What would make you reject deployment?

Unstable future-period performance, poor critical-segment recall, unmanageable alert volume, unreliable telemetry, unclear label definition, no human fallback, or inability to monitor and roll back.

## 10. What would you monitor first?

Data freshness and missingness, score and alert distributions, latency and errors, override behavior, delayed-label recall and precision, calibration, and critical-service segment performance.

## 11. How would you compare a tree ensemble?

Use the same temporal split, preprocessing discipline, metrics, calibration, threshold policy, latency, cost, and segment analysis. Approve it only if incremental operational value justifies complexity.

## 12. How would you explain a false negative to an executive?

Describe the missed incident, whether the score or threshold caused the miss, data limitations, business impact, immediate control, and the specific prevention action without overstating certainty.

## 13. How do human overrides improve the system?

They provide safety and can reveal missing features or policy mismatches. Override reasons must be structured, audited, and reviewed rather than treated as automatically correct labels.

## 14. How would you scale across business units?

Establish common platform controls and evaluation standards while validating local data definitions, prevalence, costs, capacity, and segment behavior. Avoid assuming one global threshold fits every workflow.

## 15. What is the Director-level outcome?

A governed decision system with measurable business value, reliable operations, accountable ownership, and a staged path from advisory ranking to any higher level of automation.
