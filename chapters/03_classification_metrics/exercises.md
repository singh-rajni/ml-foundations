# Classification Metrics Exercises

## Calculation

For TP=18, FP=42, FN=2, TN=938:

1. Calculate accuracy.
2. Calculate precision.
3. Calculate recall.
4. Calculate specificity.
5. Calculate F1.
6. Explain whether the result is acceptable under two different business cost assumptions.

## Coding

1. Implement confusion counts with NumPy.
2. Implement precision, recall, specificity, and F1 without scikit-learn.
3. Generate ROC and precision-recall curves.
4. Compare your functions with scikit-learn.
5. Calculate metrics for at least five thresholds.

## Analysis

1. Keep ranking scores fixed but increase negative examples. Observe ROC and PR behavior.
2. Compare accuracy with balanced accuracy.
3. Report absolute false alerts per day rather than only a false positive rate.
4. Evaluate metrics by `service_criticality` and `recent_deploy`.

## Director case

The model has ROC-AUC 0.93 and average precision 0.31 on a dataset with 2 percent positives. At the proposed threshold, recall is 0.90 but precision is 0.08, producing 500 alerts each day. Prepare a decision memo that states what additional information is required and what policy changes you would test.
