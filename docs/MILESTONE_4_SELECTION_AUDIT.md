# Milestone 4 Baseline-Selection Audit

## 1. Audit status

**PASS**

The development-and-validation selection artifact conforms materially to the
pre-registered Milestone 4 plan and contains 12 complete frozen
specifications. No frozen-test result is present.

## 2. Inputs inspected

This read-only audit used only:

- `docs/MILESTONE_4_BASELINE_PLAN.md`;
- `artifacts/milestone_4/baseline_selection.json`;
- `artifacts/milestone_3/full_season_separation_summary.json`; and
- `docs/MILESTONE_3_SEPARATION_RESULT.md`.

No raw data, weekly tracking partition, fitted model, or individual prediction
was inspected.

## 3. Data-access boundary

- Processed weeks are exactly `2023_w01` through `2023_w15`, once each and in
  order.
- Development is exactly Weeks 01–12.
- Validation is exactly Weeks 13–15.
- Recorded frozen-test weeks accessed: `0`.
- The artifact contains no Week 16–18 identifier, count, or metric.
- Development and validation game/play overlap is `0`.

## 4. Selection inventory

The artifact contains exactly 12 unique frozen units:

```text
3 horizons × 2 tasks × 2 populations = 12
```

The horizons are 5, 10, and 15; the tasks are regression and classification;
and the populations are all pairs and nearest observed defender. There are no
missing, duplicate, or additional units. Horizons and populations remain
separate.

## 5. Targets and populations

The binding target definitions are:

```text
regression target = separation_change
separation_change = separation_future - separation_origin

closing = 1 when separation_change < 0
closing = 0 otherwise
```

The artifact records matching regression/classification task units,
`target_horizon_matching: true`, and horizon-separated counts. Its split-level
closing rates reproduce the corresponding Milestone 3 rates.

The primary population contains all valid observed target–defender pairs. The
nearest-observed-defender population is separately selected and remains a
secondary geometric reduction, not an official coverage assignment.

## 6. Feature and leakage audit

The registered and actual feature lists are identical:

1. `separation_origin`
2. `dx`
3. `dy`
4. `abs_dx`
5. `abs_dy`
6. `target_x_origin`
7. `target_y_origin`
8. `defender_x_origin`
9. `defender_y_origin`
10. `valid_observed_defender_count_origin`
11. `defender_rank_origin`
12. `nearest_observed_defender_indicator`

Single-feature linear and logistic candidates use only
`separation_origin`. Constants use no feature.

The model feature lists contain no future coordinate or separation,
future-availability field, output-phase field, trajectory length, exclusion
field, week, split, game/play/player identifier, outcome, coverage label, or
betting field. The diagnostics report zero future fields and zero
identifier/split/week fields in model matrices.

Target and defender position/role variables were omitted because they were not
directly carried by the reused pair result and an additional join was not
authorized.

## 7. Candidate-grid audit

Every regression inventory contains exactly:

- training-mean constant;
- training-median constant;
- single-feature linear regression;
- multivariable OLS; and
- ridge at `alpha` 0.1, 1.0, and 10.0.

Every classification inventory contains exactly:

- constant development closing-rate probability;
- single-feature logistic regression; and
- multivariable logistic at `C` 0.1, 1.0, and 10.0.

No unregistered estimator or hyperparameter appears.

## 8. Selection and tie-break audit

Regression uses validation MAE and classification uses validation log loss.
For each unit, the recorded selection has the lowest candidate value for its
primary metric. Secondary metrics did not determine a selection.

The recorded regression preference is constants, single-feature linear,
multivariable OLS, then ridge; ridge ties prefer `alpha=10.0`, then 1.0, then
0.1. Classification prefers constant, single-feature logistic, then
multivariable logistic, with `C=0.1`, then 1.0, then 10.0 within the
multivariable grid.

All 12 `exact_tie` diagnostics are `false`; no tie rule was invoked.

## 9. Preprocessing audit

Every learned frozen specification records:

- numeric median imputation fitted on development training only;
- numeric standardization fitted on development training only;
- no categorical preprocessing because optional categoricals were omitted;
  and
- pipeline fitting on `development_train`.

The candidate estimator and preprocessing specification are stored together
for reconstruction under the binding pipeline contract. The artifact reports
development-only preprocessing and `validation_targets_used_for_fit: false`.
No target encoding or feature selection is registered.

## 10. Count and reconciliation audit

| Horizon | Development pairs / plays | Validation pairs / plays |
|---:|---:|---:|
| H5 | 21,216 / 8,648 | 5,211 / 2,085 |
| H10 | 13,488 / 4,837 | 3,499 / 1,245 |
| H15 | 5,320 / 1,722 | 1,620 / 502 |

These all-pair counts exactly match the Milestone 3 full-season split results.
The artifact also records:

- reconciliation mismatches: `0`;
- duplicate sample keys: `0`;
- cross-split game/play overlap: `0`;
- deterministic sample ordering: `true`;
- feature allowlist exact: `true`; and
- leakage validation: **PASS**.

## 11. Twelve frozen selections

`F_all` denotes the exact 12-feature allowlist in Section 6. Every selected
model uses `F_all`. All records share selection timestamp
`2026-07-30T23:22:41.354276Z`.

| Population | H | Task | Candidate | Hyperparameter | Features | Primary validation metric |
|---|---:|---|---|---|---|---:|
| All pairs | 5 | Regression | `multivariable_ols` | none | `F_all` | MAE 1.215534669 |
| All pairs | 10 | Regression | `multivariable_ols` | none | `F_all` | MAE 2.424871529 |
| All pairs | 15 | Regression | `ridge_alpha_10` | `alpha=10.0` | `F_all` | MAE 3.533330336 |
| Nearest observed defender | 5 | Regression | `multivariable_ols` | none | `F_all` | MAE 0.829609478 |
| Nearest observed defender | 10 | Regression | `ridge_alpha_10` | `alpha=10.0` | `F_all` | MAE 1.457673410 |
| Nearest observed defender | 15 | Regression | `multivariable_ols` | none | `F_all` | MAE 1.745489940 |
| All pairs | 5 | Classification | `multivariable_logistic_c_10` | `C=10.0` | `F_all` | Log loss 0.632433383 |
| All pairs | 10 | Classification | `multivariable_logistic_c_10` | `C=10.0` | `F_all` | Log loss 0.628722632 |
| All pairs | 15 | Classification | `multivariable_logistic_c_0.1` | `C=0.1` | `F_all` | Log loss 0.616861246 |
| Nearest observed defender | 5 | Classification | `multivariable_logistic_c_10` | `C=10.0` | `F_all` | Log loss 0.602270781 |
| Nearest observed defender | 10 | Classification | `multivariable_logistic_c_10` | `C=10.0` | `F_all` | Log loss 0.611282516 |
| Nearest observed defender | 15 | Classification | `multivariable_logistic_c_10` | `C=10.0` | `F_all` | Log loss 0.665049329 |

## 12. Comparator improvements

Improvement is comparator primary metric minus selected primary metric.
Positive values mean lower validation error for the selected model.

| Population | H | Regression comparator | MAE improvement | Classification comparator | Log-loss improvement |
|---|---:|---|---:|---|---:|
| All pairs | 5 | Single-feature linear | 0.095332 | Single-feature logistic | 0.042597 |
| All pairs | 10 | Single-feature linear | 0.218040 | Single-feature logistic | 0.037453 |
| All pairs | 15 | Median constant | 0.454552 | Single-feature logistic | 0.044833 |
| Nearest observed defender | 5 | Single-feature linear | 0.117560 | Single-feature logistic | 0.069351 |
| Nearest observed defender | 10 | Single-feature linear | 0.192787 | Single-feature logistic | 0.065577 |
| Nearest observed defender | 15 | Median constant | 0.045211 | Single-feature logistic | 0.022608 |

All 12 selected models improve on the strongest applicable constant or
single-feature comparator on validation. This is not evidence about frozen-test
performance.

## 13. Frozen-specification completeness

Every frozen record contains:

- a candidate name;
- an exact hyperparameter or explicit null where none applies;
- the complete feature subset;
- the development-only preprocessing specification;
- the primary validation metric and value; and
- a UTC selection timestamp.

The later evaluator can reconstruct each registered pipeline without making a
new model-selection or hyperparameter choice. No fitted object is required.

## 14. Prohibited-output audit

The JSON contains no:

- frozen-test metric or Week 16–18 count;
- individual prediction or pair-level record;
- player name;
- fitted coefficient;
- serialized model; or
- absolute local path.

Its only artifact path is project-relative.

## 15. Claim boundary

The recorded claims are limited to future separation dynamics for
competition-designated target–defender pairs. The artifact explicitly
disclaims official coverage responsibility, quarterback decision quality, pass
completion probability, causal defensive effectiveness, complete
passing-window openness, and betting value.

## 16. Final authorization decision

**PASS — AUTHORIZE ONE-TIME FROZEN EVALUATION**

Authorization is limited to one evaluation of the 12 frozen specifications.
It does not authorize further model selection, feature revision, tuning, or
inspection-driven changes.
