# ML Foundations for AI Leadership

A hands-on, interview-focused learning path covering regression, classification, model evaluation, calibration, regularization, and an end-to-end incident-priority classifier.

This repository follows one rule:

> Understand the intuition, derive the mathematics, implement the core idea, validate it against a trusted library, and explain the business decision.

## Book formats

- [Week 1 quick-start DOCX](book/Week_01_ML_Foundations_Study_Guide.docx)
- [Complete seven-chapter Markdown study guide](book/Week_1_ML_Foundations_Study_Guide.md)

## Seven-chapter learning path

| Chapter | Topic | Primary output |
|---|---|---|
| 01 | [Linear regression, MSE, and gradient descent](chapters/01_linear_regression/README.md) | NumPy implementation and convergence diagnostics |
| 02 | [Logistic regression, sigmoid, and cross-entropy](chapters/02_logistic_regression/README.md) | NumPy classifier compared with scikit-learn |
| 03 | [Precision, recall, F1, ROC-AUC, and PR-AUC](chapters/03_classification_metrics/README.md) | Metric selection for imbalanced incidents |
| 04 | [Threshold selection and calibration](chapters/04_thresholds_and_calibration/README.md) | Cost-aware threshold and reliability plots |
| 05 | [Bias, variance, L1/L2 regularization, and learning curves](chapters/05_bias_variance_regularization/README.md) | Generalization diagnosis and regularized models |
| 06 | [Incident-priority classifier capstone](chapters/06_incident_priority_capstone/README.md) | Reproducible end-to-end ML project |
| 07 | [25-question quiz and five-minute presentation](chapters/07_quiz_and_presentation/README.md) | Interview rehearsal and mastery assessment |

## Personalized AI Director preparation

Use the [AI Director Interview Bridge](AI_DIRECTOR_INTERVIEW_BRIDGE.md) to connect each chapter to enterprise architecture, GenAI, incident management, MLOps, and organizational leadership stories.

## Recommended order

Complete one chapter per day for seven days, then repeat the interview questions and capstone during a second week.

1. Read the chapter README without running code.
2. Reproduce the key equations on paper.
3. Run the notebook from top to bottom.
4. Close the notebook and reimplement the core function from memory.
5. Answer the interview questions aloud.
6. Write one paragraph connecting the topic to an enterprise decision.

## Recommended study material

See [RESOURCES.md](RESOURCES.md) for a chapter-by-chapter sequence using Google ML Crash Course, official NumPy and scikit-learn documentation, and selected books.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Run the full reference workflow:

```bash
python scripts/run_reference_workflow.py
```

## Repository map

```text
ml-foundations/
|-- README.md
|-- ROADMAP.md
|-- book/
|-- chapters/
|   |-- 01_linear_regression/
|   |-- 02_logistic_regression/
|   |-- 03_classification_metrics/
|   |-- 04_thresholds_and_calibration/
|   |-- 05_bias_variance_regularization/
|   |-- 06_incident_priority_capstone/
|   `-- 07_quiz_and_presentation/
|-- data/
|-- src/ml_foundations/
|-- tests/
`-- scripts/
```

## What mastery means

You are ready to move beyond this module when you can:

- Derive linear and logistic regression gradients.
- Implement both algorithms with NumPy.
- Explain why a probability threshold is a business policy, not a universal constant.
- Choose metrics for a rare high-impact event.
- Diagnose bias, variance, leakage, poor calibration, and drift.
- Present the capstone to engineers and executives using different levels of detail.

## Source notes

The code uses NumPy, pandas, matplotlib, and scikit-learn. The chapter references point to primary documentation and original papers where appropriate.
