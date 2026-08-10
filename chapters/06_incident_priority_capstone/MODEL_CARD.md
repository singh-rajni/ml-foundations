# Model Card - Synthetic Incident P1 Prioritization

## Model details

- Model family: regularized binary logistic regression.
- Preprocessing: standardization of numeric and binary features in a scikit-learn pipeline.
- Optional post-processing: probability calibration.
- Output: probability that an incident is `is_p1 = 1`.
- Intended action: prioritize human review.

## Intended use

Assist incident-management teams in ranking new incidents for review. The model is not approved for autonomous remediation, final priority assignment, or suppression of monitoring alerts.

## Out-of-scope use

- Real production use without retraining and validation on approved operational data.
- Employee performance evaluation.
- Automated disciplinary decisions.
- Use on services or regions not represented in validation.
- Decisions without human override and fallback.

## Training and evaluation data

Synthetic, time-ordered incidents generated for education. It contains no employer or client records.

## Metrics

The executed notebook records ROC-AUC, average precision, log-loss, Brier score, threshold metrics, expected cost, and segment performance. Exact values may change with package versions but should remain reproducible under the pinned random seed.

## Ethical and operational considerations

- Historical priority labels may encode inconsistent human practice.
- Different teams may have different escalation behavior.
- Low-quality monitoring can create unequal error rates across services.
- False negatives can delay response to serious incidents.
- False positives can overload teams and reduce trust.

## Limitations

- Synthetic data cannot establish production accuracy.
- The generator is simpler than a real incident ecosystem.
- The model assumes stable feature definitions and timely telemetry.
- Calibration can change under prevalence and policy shift.
- The cost function is illustrative.

## Monitoring and maintenance

Monitor data freshness, feature distributions, score distribution, threshold, alert volume, delayed-label metrics, calibration, segment behavior, overrides, latency, availability, and fallback usage.

## Approval recommendation

Educational use: approved.

Production use: not approved. Require data governance, label review, security review, privacy review, live shadow evaluation, human-in-the-loop pilot, operational ownership, and rollback testing.
