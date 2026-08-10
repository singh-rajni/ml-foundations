# Threshold and Calibration Exercises

## Calculation

Given validation probabilities and labels:

```text
p = [0.90, 0.70, 0.55, 0.40, 0.20, 0.10]
y = [1,    0,    1,    0,    0,    1]
```

1. Build confusion matrices at thresholds 0.5 and 0.3.
2. Calculate precision and recall at each threshold.
3. With false-positive cost 1 and false-negative cost 10, calculate total cost.
4. State which threshold you would select and what uncertainty remains.

## Coding

1. Sweep thresholds from 0.01 to 0.99.
2. Plot precision, recall, alert count, and expected cost.
3. Create a reliability diagram from scratch.
4. Calculate Brier score and binned ECE.
5. Compare raw and calibrated probabilities using scikit-learn.

## Analysis

1. Tune a threshold under three false-negative cost assumptions.
2. Add a minimum recall constraint.
3. Compare a global threshold with a top-k daily queue.
4. Evaluate calibration separately for critical and non-critical services.

## Director case

Operations can review 120 alerts per day. The cost model suggests a threshold producing 180 alerts, while the capacity-constrained threshold reduces P1 recall from 94 percent to 86 percent. Prepare a proposal covering staffing, staged automation, human review, data improvement, and executive risk acceptance.
