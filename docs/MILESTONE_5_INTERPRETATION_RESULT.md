# Milestone 5 Interpretation and Robustness Result

## 1. Milestone status

Milestone 5 interpretation and robustness analysis is complete.

**LIMITED ROBUSTNESS — SIGNAL IS REPRODUCIBLE BUT PERFORMANCE VARIES BY GEOMETRIC REGIME**

- Weeks 01–12 were used for fitting.
- Weeks 13–15 were used for diagnostics.
- Weeks 16–18 remained locked.
- Leakage validation: `PASS`.
- Reconciliation mismatches: `0`.
- No frozen-test rerun occurred.

The Milestone 4 out-of-sample `GO` remains valid: origin geometry contains reproducible predictive information. Validation robustness is strongest below 15 yards of origin separation. Long-separation buckets show comparator reversals and probability overprediction, while every H15 bucket has limited support. The models therefore should not be interpreted as uniformly reliable across all geometric regimes.

## 2. Analyses completed

The milestone completed:

1. standardized coefficient stability;
2. fixed feature-group ablations;
3. classifier calibration diagnostics; and
4. fixed origin-separation-bucket error analysis.

All diagnostics concern the all-pair population of competition-designated target–defender pairs.

## 3. Coefficient stability

The largest full-development standardized coefficients were:

| Task | Horizon | Recurring high-magnitude coefficients |
|---|---:|---|
| Regression | 5 | `dx` -0.502155; `separation_origin` -0.480567; `abs_dx` +0.289889; defender rank -0.286372; nearest-observed-defender indicator -0.281468 |
| Regression | 10 | `separation_origin` -1.328486; nearest indicator -1.116942; `dx` -0.737132; defender rank -0.667129; `abs_dx` +0.342283 |
| Regression | 15 | nearest indicator -2.244571; `dx` -1.386080; `abs_dy` -1.273154; `separation_origin` -1.096115; `abs_dx` -0.725823 |
| Classification | 5 | `dx` +0.600210; `separation_origin` +0.411790; nearest indicator +0.386434; defender rank +0.352470; `abs_dx` -0.186636 |
| Classification | 10 | `separation_origin` +0.782909; nearest indicator +0.660880; `dx` +0.444099; defender rank +0.320597; `abs_dx` -0.146567 |
| Classification | 15 | nearest indicator +0.772957; `dx` +0.552073; `abs_dy` +0.428025; `separation_origin` +0.351532; `abs_dx` +0.311796 |

Most dominant coefficients had sign stability near `1.00`. This recurring pattern supports a role for initial separation, signed and absolute relative geometry, and defender context in the fitted relationships.

The features are correlated, so standardized magnitude is not standalone feature importance. Coefficient signs do not imply causality. Lateral-location or direction terms were less stable: instability appeared for target y location, defender y location, and `dy` in different task–horizon combinations. Those terms should not receive substantive interpretation.

## 4. Feature-group ablations

Differences are ablation minus full-model validation metrics; positive values indicate degradation.

### Regression

| Horizon | Without field location ΔMAE | Without relative vector ΔMAE | Without defender context ΔMAE | Separation only ΔMAE |
|---:|---:|---:|---:|---:|
| 5 | +0.000890 | +0.018404 | +0.020180 | +0.095332 |
| 10 | -0.000796 | +0.008939 | +0.122387 | +0.218040 |
| 15 | +0.000067 | -0.004027 | +0.270211 | +0.477322 |

### Classification

Each cell reports `Δ log loss / Δ Brier score`.

| Horizon | Without field location | Without relative vector | Without defender context | Separation only |
|---:|---:|---:|---:|---:|
| 5 | +0.000035 / -0.000017 | +0.005583 / +0.001516 | +0.006964 / +0.004123 | +0.042600 / +0.023210 |
| 10 | -0.000145 / -0.000079 | +0.001174 / +0.000158 | +0.016279 / +0.008606 | +0.037457 / +0.019520 |
| 15 | +0.000334 / +0.000208 | -0.002047 / +0.000176 | +0.020217 / +0.010840 | +0.044769 / +0.023587 |

`separation_only` degraded validation performance most in all six models. Initial separation is therefore important but insufficient. Removing defender context caused the next-largest degradation, especially at H10 and H15. Removing absolute field-location coordinates had negligible or occasionally slightly favorable effects. Relative geometry and defender context appear to carry more predictive value than absolute field location, without implying an independently causal role for any feature.

## 5. Calibration diagnostics

| Horizon | ECE | MCE | Intercept | Slope | 95% interval for mean-probability bias |
|---:|---:|---:|---:|---:|---|
| 5 | 0.047585 | 0.151410 | -0.077330 | 0.857242 | [0.011290, 0.037854] |
| 10 | 0.034315 | 0.108265 | -0.092563 | 0.835214 | [0.007468, 0.040894] |
| 15 | 0.041700 | 0.119870 | -0.148757 | 0.804789 | [0.007488, 0.051692] |

H10 had the lowest ECE and MCE. All calibration intercepts were negative, all slopes were below the ideal reference of `1`, and mean predicted closing probabilities exceeded observed closing rates. This combination is consistent with probabilities that were somewhat too extreme on validation. H15 slope uncertainty was widest because its cohort was smaller.

The classifiers still improved the registered log-loss and Brier comparators in the Milestone 4 evaluation. Their unmodified probabilities are directionally useful, but they should not be treated as perfectly calibrated.

## 6. Reliability-bin findings

The largest absolute fixed-bin gaps were:

| Horizon | Probability bin | Samples | Calibration gap |
|---:|---|---:|---:|
| 5 | [0.9, 1.0] | 243 | -0.151410 |
| 5 | [0.3, 0.4) | 397 | -0.132002 |
| 5 | [0.8, 0.9) | 413 | -0.100926 |
| 10 | [0.8, 0.9) | 283 | -0.108265 |
| 10 | [0.1, 0.2) | 50 | -0.088473 |
| 10 | [0.9, 1.0] | 184 | -0.053847 |
| 15 | [0.8, 0.9) | 108 | -0.119870 |
| 15 | [0.7, 0.8) | 118 | -0.084962 |
| 15 | [0.9, 1.0] | 115 | -0.074088 |

The largest gaps were negative, meaning observed closing rates were below mean predicted probabilities. The clearest high-probability examples were H5 `[0.9,1.0]`, H10 `[0.8,0.9)`, and H15 `[0.8,0.9)`. The H10 `[0.1,0.2)` result had only 50 samples and should be interpreted accordingly.

## 7. Error heterogeneity by origin separation

Differences are selected model minus comparator; negative values favor the selected model.

| Horizon | Bucket | Support | Regression ΔMAE | Classification Δ log loss |
|---:|---|---|---:|---:|
| 5 | [0,3) | adequate | -0.089909 | -0.023911 |
| 5 | [3,5) | adequate | -0.131929 | -0.074431 |
| 5 | [5,10) | adequate | -0.124535 | -0.068709 |
| 5 | [10,15) | adequate | -0.119385 | -0.046480 |
| 5 | [15,20) | limited | +0.072890 | +0.050899 |
| 5 | [20,+inf) | limited | +0.256480 | +0.168667 |
| 10 | [0,3) | adequate | -0.435115 | -0.030970 |
| 10 | [3,5) | limited | -0.339179 | -0.078043 |
| 10 | [5,10) | adequate | -0.223472 | -0.061961 |
| 10 | [10,15) | adequate | -0.162726 | -0.030040 |
| 10 | [15,20) | limited | +0.015189 | +0.027332 |
| 10 | [20,+inf) | limited | +0.392032 | +0.066686 |
| 15 | [0,3) | limited | +0.192890 | -0.048368 |
| 15 | [3,5) | limited | -0.770844 | -0.096409 |
| 15 | [5,10) | limited | -0.541757 | -0.076786 |
| 15 | [10,15) | limited | -0.294161 | -0.045040 |
| 15 | [15,20) | limited | -0.678011 | +0.012249 |
| 15 | [20,+inf) | limited | -2.040370 | +0.043584 |

Selected models generally beat their comparators below 15 yards. Every adequate-support interval at H5 and H10 favored the selected model for regression MAE, classification log loss, and classification Brier score.

Both tasks showed comparator reversals at H5 and H10 in `[15,20)` and `[20,+inf)`. H15 regression reversed in `[0,3)`, while H15 classification reversed in `[15,20)` and `[20,+inf)`. No bucket was sparse, but every H15 bucket was limited. These limited-support reversals are diagnostics, not statistical-significance claims.

## 8. Threshold behavior

At the fixed `0.5` threshold:

| Horizon | Bucket | Probability bias | False-positive rate | False-negative rate |
|---:|---|---:|---:|---:|
| 5 | [0,3) | +0.012405 | 0.213894 | 0.628571 |
| 5 | [20,+inf) | +0.137491 | 0.961538 | 0.009901 |
| 10 | [0,3) | -0.003146 | 0.070461 | 0.842282 |
| 10 | [20,+inf) | +0.069464 | 1.000000 | 0.009009 |
| 15 | [0,3) | +0.000157 | 0.081340 | 0.765060 |
| 15 | [20,+inf) | +0.040099 | 1.000000 | 0.010309 |

False-negative rates were high in `[0,3)`, while false-positive rates became extreme in `[20,+inf)`. H10 and H15 reached a false-positive rate of `1.000` in the longest-separation bucket. Closing probabilities were positively biased in both long-separation buckets at every horizon. A single fixed threshold therefore does not behave uniformly across origin-separation regimes; this report does not optimize or recommend a replacement threshold.

## 9. Consolidated interpretation

The supported interpretation is:

- origin separation provides the strongest basic signal but is insufficient alone;
- signed and absolute relative geometry add useful information;
- defender context adds substantial medium- and long-horizon information;
- absolute field location contributes little in these registered linear models;
- average validation performance generalizes, but robustness is weaker when the target and observed defensive entity begin very far apart; and
- probability estimates are directionally useful but somewhat overconfident.

These patterns concern target–defender geometry and do not identify official coverage behavior.

## 10. Relationship to Milestone 4

Milestone 4 remains `GO`, and its frozen evaluation is unchanged. Milestone 5 used development and validation data only, did not alter the frozen selections, and did not rerun Weeks 16–18. Those weeks must not be reused for new selection.

Milestone 5 narrows the appropriate claim from uniform predictive performance to:

> reproducible average predictive signal with meaningful geometric heterogeneity

## 11. Validation evidence

- Coefficient stability: `200` play-level bootstrap resamples per model.
- Calibration uncertainty: `500` play-level resamples per horizon.
- Adequate-bucket error uncertainty: `300` play-level resamples.
- Bootstrap seed: `2026`.
- Validation rows used for fitting: `0`.
- Duplicate pair-horizon keys: `0`.
- Leakage validation: `PASS`.
- Reconciliation mismatches: `0`.
- No individual predictions, probabilities, pair rows, fitted models, or pipelines were persisted.

## 12. Claim boundary

The findings concern competition-designated target–defender pairs and observed defensive entities. They do not establish:

- official coverage assignments;
- quarterback decision quality;
- pass completion probability;
- causal feature effects;
- complete passing-window openness;
- calibrated probabilities in another season;
- betting value; or
- production or deployment readiness.

## 13. Final decision

**LIMITED ROBUSTNESS — PROCEED WITH CAUTION**

The predictive signal is real relative to the registered baselines, and performance is strongest in commonly represented separation ranges. Long-separation performance, probability overprediction, and threshold behavior require caution. Validation calibration is adequate for continued research diagnostics, but not strong enough for confident probability interpretation across all geometric regimes.

## 14. Recommended next milestone

The next milestone should be **project consolidation and portfolio release**, not further exploratory model tuning:

1. document the architecture and data contracts;
2. add model cards and limitations;
3. create one reproducible demo using development/validation artifacts only;
4. add release verification and CI smoke checks;
5. preserve the frozen-test result as final; and
6. avoid further model selection against Weeks 16–18.

These tasks are recommendations only and are not implemented here.
