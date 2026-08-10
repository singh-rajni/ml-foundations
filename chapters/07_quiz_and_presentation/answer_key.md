# Answer Key

## 1.

$\hat{y}=wx+b$. Here $x$ is the feature, $w$ the slope, $b$ the intercept, and $\hat{y}$ the prediction. For multiple features use $\hat{y}=X\theta$.

## 2.

Residuals are `[1, 1, -2]`. Squared residuals are `[1, 1, 4]`. MSE is `6/3 = 2`. RMSE is $\sqrt{2}\approx1.414$. MAE is `(1+1+2)/3 = 1.333`.

## 3.

$$
\frac{\partial J}{\partial w}=\frac{2}{n}\sum(\hat{y}_i-y_i)x_i
$$

$$
\frac{\partial J}{\partial b}=\frac{2}{n}\sum(\hat{y}_i-y_i)
$$

## 4.

Possible causes include an excessive learning rate, unscaled extreme features, invalid values, overflow, wrong gradient sign, and shape broadcasting errors. Check loss and gradient magnitude per step, finite values, feature ranges, shapes, and behavior on a tiny known dataset.

## 5.

Require business impact of the RMSE difference, confidence intervals, segment behavior, latency, capacity, reliability, calibration if relevant, explainability, infrastructure and support cost, adoption impact, rollout risk, and whether the improvement changes a decision.

## 6.

Odds are $p/(1-p)$. Log-odds or logit are $\log(p/(1-p))$. Logistic regression models log-odds as a linear function of features and maps the result through sigmoid to probability.

## 7.

$\sigma(0)=0.5$. A zero linear score maps to the default 0.5 probability boundary, but the operational threshold need not be 0.5.

## 8.

For one case: $-[y\log(p)+(1-y)\log(1-p)]$. If the model gives a probability near zero to a true positive, the loss becomes very large.

## 9.

$$
\nabla J=\frac{1}{n}X^T(p-y)
$$

## 10.

Check feature scaling, intercept, regularization, `C`, solver, convergence tolerance, iteration limit, label mapping, train split, sample weighting, and numerical stability.

## 11.

Total is 1,000. Accuracy is 0.956. Precision is `18/60 = 0.30`. Recall is `18/20 = 0.90`. Specificity is `938/980 = 0.9571`. F1 is `2*0.3*0.9/(0.3+0.9) = 0.45`.

## 12.

If only 2 percent are P1, predicting every case as non-P1 produces 98 percent accuracy and zero recall.

## 13.

ROC plots true positive rate against false positive rate. Precision-recall plots precision against recall. Each point corresponds to a threshold.

## 14.

It focuses on positive-class retrieval and exposes the precision cost of increasing recall. ROC false-positive rate can look small even when the absolute number of false alerts is large.

## 15.

Positive prevalence, daily incident volume, alert count, review capacity, false-negative impact, false-positive burden, time saved, safety requirements, segment performance, and trust determine acceptability.

## 16.

The threshold depends on costs, prevalence, capacity, safety, product policy, and the required intervention. It is separate from model training.

## 17.

Use validation data or cross-validation. Freeze preprocessing, model, calibration, and threshold before evaluating once on the locked test set.

## 18.

The model is overconfident in that group: predicted probability is 0.8 while observed frequency is 0.5. Check sample size, population shift, and calibration by segment.

## 19.

Brier score is mean squared error between probability and binary outcome. It mixes calibration and discrimination and does not show where calibration fails.

## 20.

Likely high bias, weak features, bad labels, or excessive regularization. Test richer features, interactions, a more expressive model, label improvement, and reduced regularization.

## 21.

Leakage, duplicates across splits, temporal or group shift, inconsistent preprocessing, unstable labels, small validation sample, or evaluation code errors.

## 22.

L1 uses absolute magnitude and can create exact zeros; L2 uses squared magnitude and shrinks smoothly. L1 may support sparse deployment but can be unstable among correlated features. L2 often improves stability.

## 23.

Examples: final priority, final resolver team, time to resolution, post-incident impact, escalation timestamp, root cause, final status, or remediation outcome.

## 24.

Offline evaluation and review; live shadow mode; limited human-in-the-loop pilot; controlled canary expansion with monitoring and rollback. Mature operation follows only after acceptance criteria are met.

## 25.

Protect operations first and ensure the incident follows standard escalation. Pause or restrict the pilot if required. Reconstruct model version, data, score, threshold, and workflow. Determine whether the miss came from unavailable data, ranking, calibration, threshold, label ambiguity, or system failure. Communicate impact and facts without speculation. Implement the specific control, add a regression test, review similar cases, and update rollout criteria before resuming.
