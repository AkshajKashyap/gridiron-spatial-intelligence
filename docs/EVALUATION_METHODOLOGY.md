# Evaluation Methodology

## 1. Source and cohort reconciliation

The structural audit reconciled 18 weekly input/output pairs, 4,880,579 input
rows, 562,936 output rows, 272 games, and 14,108 tracked plays. Cohort
construction freezes six tables plus one exclusion ledger. Every eligible or
excluded unit reconciles by source table and deterministic primary reason;
duplicate keys, split conflicts, missing joins, malformed sequences, and
coordinate failures are hard errors.

Descriptive geometry can use every valid observed input defender. Future
evaluation uses only defenders observed at origin with valid supplied output
through the requested horizon.

## 2. Predetermined temporal split

| Split | Weeks | Permitted use |
|---|---|---|
| Development | 01–12 | Fit preprocessing and models |
| Validation | 13–15 | Select preregistered candidates and run diagnostics |
| Frozen test | 16–18 | One final evaluation after selections were audited |

Games and all related rows remain in one chronological split. Random
row-, frame-, pair-, or play-level splitting is prohibited.

## 3. Horizon-specific samples

H5, H10, and H15 are separate tasks. A pair is eligible only when target and
defender have contiguous valid output through that horizon; missing output is
not imputed into a label. The full-season descriptive result contains 31,937
H5, 20,680 H10, and 8,539 H15 future-evaluable pairs, totaling 61,156. The
shrinking H15 cohort is interpreted as more selective.

## 4. Regression and classification targets

```text
separation_change = separation_future - separation_origin
closing = 1 if separation_change < 0 else 0
```

Regression predicts signed separation change. Classification predicts whether
the same pair closes over the horizon. The zero classification threshold was
fixed and not tuned.

## 5. Registered baselines

Regression candidates were training mean/median, separation-only OLS,
multivariable OLS, and multivariable ridge at three registered alphas.
Classification candidates were the training closing-rate probability,
separation-only logistic regression, and multivariable L2 logistic regression
at three registered C values. No unregistered model family or automated
search entered selection.

## 6. Validation-selection procedure

Candidates and preprocessing fit only on Weeks 01–12. Selection was independent
by population, horizon, and task using validation MAE for regression and
validation log loss for classification. Exact ties favored the simpler model,
then stronger regularization. Secondary metrics never controlled selection.

## 7. Selection audit

Before test access, the audit verified 12 unique specifications, exact feature
allowlists, registered preprocessing and grids, deterministic populations,
zero frozen-week access, reconciliation, and comparator definitions. It
authorized exactly one frozen evaluation.

## 8. One-time frozen evaluation

The audited specifications were refit under the declared protocol and
evaluated once on Weeks 16–18. The stored result declares one execution, zero
selection/comparator changes, leakage `PASS`, and zero reconciliation
mismatches. That result is final for release `0.1.0`; it must not be rerun to
guide new choices.

## 9. Primary and secondary metrics

| Task | Primary | Secondary |
|---|---|---|
| Regression | MAE | RMSE, median absolute error, descriptive R² |
| Classification | Log loss | Brier score, ROC AUC, descriptive accuracy at 0.5 |

Counts and metrics are reported separately by split, horizon, and all-pair
versus nearest-origin-defender population.

## 10. Selected-minus-comparator convention

Every headline difference is:

```text
selected-model metric - registered-comparator metric
```

For MAE, log loss, and Brier score, negative values favor the selected model.
This sign convention is fixed across validation, frozen results, and
robustness reports.

## 11. Play-cluster bootstrap

Frozen comparisons use seed 2026, 500 resamples, and 95% intervals. A draw
resamples plays and retains all target–defender pair rows from each selected
play. This preserves within-play dependence instead of pretending pair rows
are independent. Intervals are uncertainty summaries, not formal
significance claims.

## 12. Coefficient stability

Milestone 5 resampled development plays and summarized standardized
coefficients and sign stability. It did not inspect Weeks 16–18 or change the
frozen specifications. Coefficients describe predictive associations under
correlated geometric features, not causal effects.

## 13. Feature-group ablations

Fixed groups removed separation, defender context, relative geometry, or
absolute field location from the frozen feature set. Ablations were evaluated
on validation only. Separation removal caused the largest general degradation;
defender-context removal was next, while absolute-location removal had little
or occasionally favorable effect.

## 14. Calibration diagnostics

Fixed probability bins, Brier decomposition, calibration intercept/slope, and
play-clustered intervals were computed on development/validation. All
validation intercepts were negative and slopes below 1; predicted closing
probabilities exceeded observed rates on average. These findings qualify
probability interpretation and are not a frozen-test recalibration.

## 15. Origin-separation error buckets

Fixed buckets `[0,3)`, `[3,5)`, `[5,10)`, `[10,15)`, `[15,20)`, and
`[20,+inf)` were evaluated on validation. Reports retain support flags,
regression error/bias, classification score differences, probability bias,
and confusion rates. Several long-separation buckets reversed comparator
ordering; all H15 buckets had limited support.

## 16. Decision rules

`GO` required primary validation improvement, same-direction frozen
improvement, passing leakage/reconciliation checks, and evidence not dependent
only on H15. `LIMITED GO` covered mixed or narrow evidence. `NO GO` covered no
validation improvement, clear frozen reversal, or invalidating leakage or
reconciliation failure.

Milestone 4 met `GO`. Milestone 5 did not revise that result; it concluded
`LIMITED ROBUSTNESS — PROCEED WITH CAUTION`.

## 17. Threats to validity

- The competition target is conditioned upon rather than selected by a model.
- Defensive entities and supplied futures are incomplete.
- Nearest observed defender is not official coverage responsibility.
- Longer horizons are increasingly selective.
- One 2023 chronology cannot establish cross-season generalization.
- Correlated features complicate coefficient interpretation.
- Fixed buckets reveal heterogeneity but do not prove subgroup significance.
- Raw probabilities were somewhat overconfident on validation.
- No result establishes completion probability, causality, passing-window
  openness, betting value, or production readiness.

Validation diagnostics and frozen evidence remain distinct: Milestone 5
describes Weeks 01–15 and cannot revise or supplement the one-time frozen
decision.
