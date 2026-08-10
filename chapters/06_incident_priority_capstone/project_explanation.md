# Five-Minute Project Explanation

## 0:00-0:40 - Problem

P1 incidents are rare but expensive. During busy periods, manual triage can delay the right expert response. I built a synthetic, public-safe project that estimates P1 probability at incident creation and prioritizes human review. It is intentionally advisory, not autonomous.

## 0:40-1:20 - Data and leakage controls

The dataset contains time-ordered operational signals such as service criticality, customer impact, affected services, error rate, recent deployment, recurrence, and monitoring confidence. I excluded every field created after triage. I used a 60/20/20 temporal split so validation and test periods represent later incidents.

## 1:20-2:00 - Model

I established simple baselines and trained a standardized, regularized logistic regression pipeline. I chose it as the first candidate because it is explainable, low-latency, easy to operate, and provides probabilities. The repository also contains a NumPy implementation so the optimization is transparent.

## 2:00-2:50 - Evaluation and decision policy

Because P1 incidents are rare, accuracy is not the primary metric. I evaluate average precision, ROC-AUC, log-loss, Brier score, recall, precision, alert volume, expected cost, and service-criticality segments. The model produces a probability; the validation threshold converts it into an urgent-review decision using false-negative and false-positive costs plus a recall guardrail.

## 2:50-3:30 - Calibration

I inspect reliability plots and Brier score because ranking quality does not guarantee that a score of 0.3 behaves like a 30 percent event rate. Calibration matters if probabilities are used for staffing, risk tiers, or multiple intervention levels.

## 3:30-4:20 - Production design

The system validates data, generates features, scores the incident, applies a versioned calibration and threshold policy, and sends the recommendation to the incident UI. It logs model version, features, probability, decision, override, and final label. If data or the model is unavailable, the workflow falls back to standard triage.

## 4:20-5:00 - Recommendation

I would approve shadow mode followed by a limited human-in-the-loop pilot. I would not approve autonomous remediation. Expansion requires stable live recall, manageable alert volume, acceptable calibration, no severe segment failures, named ownership, auditability, and tested rollback. The key lesson is that the production artifact is not merely a model; it is a governed decision system.
