# Milestone 4 Leakage-Safe Baseline Prediction Plan

## 1. Objective and scope

Milestone 4 is a pre-registered baseline study of whether geometry available at
the validated origin predicts future target–defender separation behavior better
than constant or single-feature baselines.

This is not a broad machine-learning benchmark. It will not add complex model
families, search for new thresholds, or use the frozen-test split to revise the
study design.

## 2. Prediction units and populations

One sample is:

```text
one target–defender pair at one predetermined horizon
```

Independent datasets and models will be built for:

- 5 frames;
- 10 frames; and
- 15 frames.

Horizons will never be combined in one model.

The primary population is all valid observed target–defender pairs. The
secondary descriptive population is the nearest observed defender at origin.
The latter is analyzed separately and is **not an official coverage
assignment**. It cannot determine the primary all-pair model.

Every horizon-specific dataset must reconcile to the validated Milestone 3
counts:

| Horizon | Expected pairs | Expected plays |
|---:|---:|---:|
| H5 | 31,937 | 12,966 |
| H10 | 20,680 | 7,404 |
| H15 | 8,539 | 2,730 |

These are different horizon-eligible populations, not repeated observations
of one universal play cohort.

## 3. Targets

### 3.1 Regression

```text
target = separation_change
separation_change = separation_future - separation_origin
```

Negative values indicate contracting separation, zero indicates unchanged
separation, and positive values indicate expanding separation.

### 3.2 Classification

The binary target is named `closing`:

```text
closing = 1 when separation_change < 0
closing = 0 otherwise
```

The zero threshold is fixed. It will not be tuned or replaced with a nonzero
threshold.

Targets must be derived only from the matching future horizon. A feature may
not depend on the target or on any future observation.

## 4. Permitted origin-only features

The minimal numeric feature set is fixed to information available at the
validated origin:

- `separation_origin`;
- signed `dx`;
- signed `dy`;
- absolute `dx`;
- absolute `dy`;
- target normalized `x`;
- target normalized `y`;
- defender normalized `x`;
- defender normalized `y`;
- valid observed-defender count at origin;
- deterministic defender rank by origin separation; and
- nearest-observed-defender indicator.

Defender rank uses ascending origin separation, with defender identifier as the
deterministic tie-breaker. The identifier determines rank but is not itself a
model feature.

Target and defender position or role may be included as optional categorical
features only if they already exist in the validated pair-building inputs
without a new or ambiguous join. Otherwise they will be omitted from the first
baseline.

## 5. Prohibited features and leakage fields

The model feature matrix must not contain:

- future coordinates;
- future separation;
- future-availability flags;
- output-phase fields;
- trajectory lengths;
- exclusion reasons encoding future availability;
- week number as a shortcut feature;
- split;
- game or play identifiers;
- player identifiers;
- target outcome information;
- pass or completion result;
- official coverage labels; or
- betting information.

Identifiers may be retained outside the feature matrix only for grouping,
reconciliation, chronological splitting, deterministic ordering, and
play-cluster resampling.

## 6. Frozen temporal split

The existing chronological split is binding:

| Split | Weeks | Use |
|---|---|---|
| `development_train` | 01–12 | Fit preprocessing and models |
| `validation` | 13–15 | Select the allowed candidate or hyperparameter |
| `frozen_test` | 16–18 | One final evaluation after freezing |

Rules:

1. Fit preprocessing only on development-training rows.
2. Fit candidate models only on development-training rows.
3. Use validation only for the pre-registered selection decisions.
4. Freeze features, preprocessing, candidate, and hyperparameters before
   accessing frozen-test outcomes.
5. Evaluate the frozen test exactly once.
6. Do not revise any study choice after viewing frozen-test results.

Random row-level splitting is prohibited.

## 7. Pre-registered baselines

### 7.1 Regression candidates

1. Training-set mean prediction.
2. Training-set median prediction.
3. Ordinary linear regression using only `separation_origin`.
4. Multivariable ordinary linear regression using the permitted features.
5. Multivariable ridge regression with:

```text
alpha in {0.1, 1.0, 10.0}
```

Ridge alpha is selected by validation MAE.

### 7.2 Classification candidates

1. Constant training closing-rate probability.
2. Logistic regression using only `separation_origin`.
3. Multivariable L2 logistic regression with:

```text
C in {0.1, 1.0, 10.0}
```

`C` is selected by validation log loss.

This milestone excludes trees, boosting, neural networks, feature selection,
polynomial expansion, automated tuning, and unregistered model families.

## 8. Preprocessing

For each horizon and task:

- impute numeric missing values with development-training medians;
- standardize numeric features using development-training statistics for
  linear and logistic models;
- one-hot encode any permitted categorical fields, ignoring unknown
  categories;
- perform no target encoding; and
- fit no preprocessing state on validation or frozen-test rows.

A scikit-learn pipeline is preferred so fitted preprocessing cannot be
separated from model fitting.

Constant baselines use their development-training target statistic and require
no feature preprocessing.

## 9. Metrics and evaluation levels

### 9.1 Regression

Primary metric:

- MAE.

Secondary metrics:

- RMSE;
- median absolute error; and
- R² as descriptive context only.

Models will not be selected by R².

### 9.2 Classification

Primary metric:

- log loss.

Secondary metrics:

- Brier score;
- ROC AUC when both classes are present; and
- accuracy at probability threshold `0.5` as descriptive context only.

Models will not be selected by accuracy.

Metrics and sample/play counts will be reported separately for:

- development training, validation, and frozen test;
- H5, H10, and H15;
- the all-pair primary population; and
- the nearest-observed-defender secondary population.

Pair rows will not be presented as statistically independent.

## 10. Play-clustered uncertainty

Final selected-model comparisons use deterministic play-cluster bootstrap
intervals:

```text
seed = 2026
resamples = 500
confidence level = 95%
```

Each bootstrap draw resamples plays and retains all target–defender pair rows
belonging to each sampled play. It does not resample individual rows.

For each split-reported final comparison, intervals will cover the metric
difference between the selected model and the strongest applicable constant
baseline:

- regression MAE difference;
- classification log-loss difference; and
- classification Brier-score difference.

The regression constant comparator is the training-mean or training-median
candidate with the lower validation MAE. Classification uses the constant
training closing-rate probability. Comparator selection occurs before
frozen-test evaluation. Negative metric differences indicate improvement.

No conventional row-independent standard errors or p-values will be reported.

## 11. Model-selection hierarchy

Selection is performed independently for every horizon and task:

1. Fit every pre-registered candidate on development training.
2. Evaluate candidates on validation.
3. Select the candidate with the best primary validation metric.
4. Resolve an exact tie in favor of the simpler candidate.
5. Freeze the complete selected pipeline.
6. Evaluate frozen test exactly once.

The simplicity order is constant baseline, single-feature model,
multivariable unregularized model, then multivariable regularized model. A tie
within a regularization grid selects stronger regularization: ridge
`alpha=10.0` or logistic `C=0.1` before weaker settings.

The nearest-observed-defender population is evaluated separately and does not
participate in primary all-pair selection.

## 12. Required leakage and reconciliation validation

All checks must pass before fitting:

- requested weeks and their split assignments match the frozen contract;
- pair-horizon keys are unique;
- the feature matrix contains zero future-derived fields;
- no `(game_id, play_id)` appears in more than one split;
- every target is derived only from its matching future horizon;
- preprocessing is fitted only on development training;
- frozen-test outcomes are inaccessible during selection;
- sample ordering is deterministic;
- all-pair samples reconcile to 31,937 H5, 20,680 H10, and 8,539 H15 pairs;
- each sample is well formed and has an explicit horizon; and
- nearest-observed-defender membership follows the frozen deterministic rule.

Unexpected duplicates, malformed samples, split overlap, count mismatches, or
leakage cause a hard failure. They are not silently dropped or repaired.

## 13. Predetermined decision rules

### GO

Choose **GO** only when:

- a selected multivariable model improves the primary validation metric over
  the strongest constant or single-feature baseline;
- frozen-test performance improves in the same direction;
- every leakage and reconciliation check passes; and
- the result is not entirely dependent on H15.

### LIMITED GO

Choose **LIMITED GO** when validation improves but frozen-test evidence is
mixed, very small, or restricted to one horizon, or when calibration and error
improvements are inconsistent.

### NO GO

Choose **NO GO** when no learned model improves the primary validation metric,
frozen-test performance clearly reverses the validation result, or a leakage
or reconciliation failure invalidates the experiment.

No decision requires statistical significance. A small improvement will not
be described as practically valuable without additional evidence.

## 14. Required later outputs

One later authorized run should produce:

- one compact JSON result;
- selected model and hyperparameter for each horizon and task;
- validation and frozen-test metrics;
- clustered uncertainty intervals;
- feature names and preprocessing configuration; and
- leakage and reconciliation diagnostics.

Individual predictions will not be persisted unless separately authorized.
This planning task creates none of these outputs.

## 15. Claim boundary

Even a successful baseline would concern only future separation dynamics for
competition-designated target–defender pairs. It would not establish:

- official coverage responsibility;
- quarterback target-selection quality;
- pass completion probability;
- causal defensive effectiveness;
- complete passing-window openness; or
- betting value.

The study evaluates a constrained descriptive-prediction question, not the
quality of a receiver, defender, quarterback, or play outcome.

## 16. Explicit non-goals

This plan does not implement a dataset, features, preprocessing, tests, models,
model selection, evaluation, or artifacts. It does not inspect model
performance or authorize work beyond the baseline study.
