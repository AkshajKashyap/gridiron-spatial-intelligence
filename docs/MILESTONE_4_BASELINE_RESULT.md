# Milestone 4 Final Baseline Result

## 1. Milestone status

Milestone 4 is complete with a final status of **GO** within the established target-centric claim boundary.

- Pre-evaluation selection audit: `PASS`
- Frozen-test evaluator executions: `1`
- Leakage validation: `PASS`
- Frozen-result reconciliation mismatches: `0`

## 2. Evaluation protocol

- Development training used Weeks 01–12.
- Weeks 13–15 were used only for validation and model selection.
- The frozen test set, Weeks 16–18, was evaluated exactly once after the selection audit passed.
- Validation rows used in the final model fit: `0`.
- Horizons, prediction tasks, and analytic populations were evaluated separately.
- The two populations were all evaluable target–defender pairs and the defender nearest to the target at the immutable origin.
- Nearest-defender status is a descriptive geometric relationship, not an official coverage assignment.
- Confidence intervals used play-clustered bootstrap resampling. Defender-pair rows from the same play were not treated as independent observations.

## 3. Frozen-test population counts

| Population | Horizon | Evaluation rows | Play clusters |
|---|---:|---:|---:|
| All evaluable pairs | 5 | 5,510 | 2,233 |
| All evaluable pairs | 10 | 3,693 | 1,322 |
| All evaluable pairs | 15 | 1,599 | 506 |
| Nearest defender | 5 | 2,161 | 2,161 |
| Nearest defender | 10 | 1,300 | 1,300 |
| Nearest defender | 15 | 501 | 501 |

The progressively smaller longer-horizon cohorts constrain interpretation, especially at horizon 15.

## 4. Frozen-test regression results

Differences are selected-model MAE minus comparator MAE, in yards. Negative values favor the selected model.

| Population | Horizon | Selected model | Comparator | Selected MAE | Comparator MAE | Difference | Play-clustered 95% interval |
|---|---:|---|---|---:|---:|---:|---|
| All evaluable pairs | 5 | OLS | Single-feature linear | 1.222541 | 1.299371 | -0.076830 | [-0.090869, -0.063151] |
| All evaluable pairs | 10 | OLS | Single-feature linear | 2.424684 | 2.635341 | -0.210657 | [-0.246583, -0.177714] |
| All evaluable pairs | 15 | Ridge, alpha 10 | Median constant | 3.388624 | 3.920724 | -0.532100 | [-0.666030, -0.404377] |
| Nearest defender | 5 | OLS | Single-feature linear | 0.848883 | 0.957888 | -0.109005 | [-0.130741, -0.090518] |
| Nearest defender | 10 | Ridge, alpha 10 | Single-feature linear | 1.492773 | 1.684460 | -0.191687 | [-0.237026, -0.153502] |
| Nearest defender | 15 | OLS | Median constant | 1.594830 | 1.602184 | -0.007354 | [-0.100430, 0.077597] |

Five of the six intervals lie wholly below zero. The nearest-defender horizon-15 result is the primary weak result: its estimated improvement is small and its interval crosses zero. These intervals are uncertainty summaries under the registered play-clustered bootstrap, not claims of formal statistical significance.

## 5. Frozen-test classification results

Differences are selected-model score minus single-feature logistic score. Negative values favor the selected model.

| Population | Horizon | Selected logistic model | Metric | Selected | Comparator | Difference | Play-clustered 95% interval |
|---|---:|---|---|---:|---:|---:|---|
| All evaluable pairs | 5 | C=10 | Log loss | 0.631936 | 0.667316 | -0.035380 | [-0.043166, -0.026323] |
| All evaluable pairs | 5 | C=10 | Brier score | 0.218763 | 0.237522 | -0.018759 | [-0.021879, -0.015035] |
| All evaluable pairs | 10 | C=10 | Log loss | 0.620705 | 0.659148 | -0.038443 | [-0.048433, -0.028602] |
| All evaluable pairs | 10 | C=10 | Brier score | 0.215105 | 0.234076 | -0.018971 | [-0.023276, -0.014679] |
| All evaluable pairs | 15 | C=0.1 | Log loss | 0.611150 | 0.661968 | -0.050818 | [-0.068510, -0.033828] |
| All evaluable pairs | 15 | C=0.1 | Brier score | 0.211042 | 0.235245 | -0.024204 | [-0.031834, -0.017275] |
| Nearest defender | 5 | C=10 | Log loss | 0.605635 | 0.667145 | -0.061510 | [-0.077355, -0.048318] |
| Nearest defender | 5 | C=10 | Brier score | 0.207371 | 0.237530 | -0.030159 | [-0.036656, -0.024615] |
| Nearest defender | 10 | C=10 | Log loss | 0.621001 | 0.671043 | -0.050041 | [-0.067371, -0.032730] |
| Nearest defender | 10 | C=10 | Brier score | 0.214282 | 0.239444 | -0.025162 | [-0.032282, -0.018131] |
| Nearest defender | 15 | C=10 | Log loss | 0.651810 | 0.687425 | -0.035615 | [-0.061128, -0.008813] |
| Nearest defender | 15 | C=10 | Brier score | 0.228913 | 0.247151 | -0.018238 | [-0.029515, -0.007301] |

The selected classifier improved both log loss and Brier score in all six population–horizon cells, and every reported interval lies below zero. Classification improvement was more uniform across the registered cells than regression improvement.

## 6. Validation-to-frozen consistency

- Frozen-test direction agreed with validation for all `12` primary comparisons.
- Direction reversals: `0`.
- Eight of the confirmations came from horizons 5 and 10, so the conclusion does not depend on the smaller horizon-15 cohorts.
- Post-audit selected-model changes: `0`.
- Post-audit comparator changes: `0`.

## 7. What was learned

Within these registered baselines, origin geometry contains reproducible information about later target–defender separation beyond a constant prediction or an origin-separation-only comparator. This result does not establish which individual features are useful: no feature-importance, coefficient-interpretation, or ablation claim is made here.

## 8. Practical magnitude

For all evaluable pairs, the regression MAE improvements were approximately `0.077`, `0.211`, and `0.532` yards at horizons 5, 10, and 15. The nearest-defender horizon-15 improvement was approximately `0.007` yards and remains uncertain. Classification log-loss improvements ranged from approximately `0.035` to `0.062`.

These are held-out predictive differences for the specified competition-target cohorts. They are not evidence of operational, commercial, betting, or deployment value.

## 9. Validation evidence

- Frozen evaluator execution count: `1`
- Compilation and focused evaluator tests: `PASS`
- Preregistered primary comparisons evaluated: `12`
- Validation rows in final fitting: `0`
- Leakage checks: `PASS`
- Result reconciliation mismatches: `0`
- Duplicate result records: `0`
- Selected-model or comparator changes after audit: `0`
- Persisted predictions, pair-level outputs, fitted models, pipelines, or coefficients: none

## 10. Claim boundary

The result concerns future separation between the competition-designated target and observed defenders. The supplied target designation is conditioned upon; it is not predicted.

It does not support claims about:

- official coverage assignments;
- quarterback target selection;
- completion probability or completion causation;
- causal defender or receiver effects;
- passing windows;
- full-field defensive control;
- betting performance; or
- deployment readiness.

## 11. Decision

**GO — ORIGIN GEOMETRY HAS REPRODUCIBLE OUT-OF-SAMPLE PREDICTIVE SIGNAL**

This decision applies to the registered simple baselines and the bounded target-centric tasks. It supports further research, not broader football, product, operational, or betting claims.

## 12. Next milestone

Proceed to controlled interpretation and robustness work:

1. Estimate standardized coefficients and their stability using development-set resamples.
2. Run feature ablations using development and validation data only.
3. Produce calibration plots for the classification tasks.
4. Slice errors by origin-separation bucket and horizon.
5. Keep Weeks 16–18 locked and do not reuse them for selection.

No next-milestone work is implemented in this report.
