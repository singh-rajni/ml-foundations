#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"

for notebook in \
  "$repo_dir/chapters/01_linear_regression/01_linear_regression_numpy.ipynb" \
  "$repo_dir/chapters/02_logistic_regression/02_logistic_regression_numpy_vs_sklearn.ipynb" \
  "$repo_dir/chapters/03_classification_metrics/03_classification_metrics.ipynb" \
  "$repo_dir/chapters/04_thresholds_and_calibration/04_thresholds_and_calibration.ipynb" \
  "$repo_dir/chapters/05_bias_variance_regularization/05_bias_variance_regularization.ipynb" \
  "$repo_dir/chapters/06_incident_priority_capstone/06_incident_priority_classifier.ipynb"
do
  chapter_dir="$(dirname "$notebook")"
  file_name="$(basename "$notebook")"
  echo "Executing $file_name"
  (cd "$chapter_dir" && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 "$file_name")
done
