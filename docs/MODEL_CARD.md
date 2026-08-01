# Model Card

## 1. Model overview

Release `0.1.0` contains registered linear and logistic baselines for a bounded
research question:

> How origin geometry for a competition-designated target and observed
> defensive entities relates to future target–defender separation dynamics.

The target is supplied by the competition data and is not predicted. Models
are fit separately for H5, H10, and H15 and for all evaluable pairs versus the
nearest observed defender at the origin.

## 2. Intended use

- Reproduce a leakage-safe baseline study of target–defender separation.
- Compare simple origin-geometry models with constant or
  origin-separation-only baselines.
- Study coefficient stability, fixed feature-group ablations, validation
  calibration, and geometric error regimes.
- Demonstrate research engineering with deterministic cohorts, temporal
  splits, and checksum-backed evidence.

## 3. Out-of-scope uses

The models are not intended for official coverage assignment, quarterback
target selection, completion probability, causal player evaluation,
full-field defensive control, passing-window estimation, betting, live
decision support, or production deployment.

## 4. Prediction units

The primary unit is one competition-designated target–defender pair at one
immutable play origin and one horizon. Pair keys include game, play, target,
defender, origin, and horizon. The all-pair population contains every
output-evaluable observed defender; the secondary population retains the
defender geometrically nearest at origin. Nearest does not imply coverage.

## 5. Targets

- Regression: `separation_change = separation_future - separation_origin`.
  Negative values indicate closing and positive values expanding separation.
- Classification: `closing = 1` when `separation_change < 0`; otherwise `0`.

Labels use only the matching supplied output horizon. Missing future
trajectories produce unavailable labels, not negative examples.

## 6. Features

Registered numeric features are origin separation; signed and absolute
target-to-defender `dx`/`dy`; normalized target and defender `x`/`y`; valid
observed-defender count; deterministic defender rank; and nearest-origin
indicator.

Game, play, player, week, split, output-phase values, trajectory length,
future availability, future coordinates, future separation, outcomes, and
coverage labels are excluded from the feature matrix.

## 7. Model families

Regression candidates comprise development mean/median constants,
origin-separation-only OLS, multivariable OLS, and ridge with
`alpha ∈ {0.1, 1.0, 10.0}`. Classification candidates comprise the
development closing-rate constant, origin-separation-only logistic
regression, and multivariable L2 logistic regression with
`C ∈ {0.1, 1.0, 10.0}`.

Median imputation and standardization are fit on development rows only.
Trees, boosting, neural networks, feature selection, polynomial expansion,
and automated tuning are outside this release.

## 8. Temporal splits

| Split | Weeks | Role |
|---|---|---|
| Development | 01–12 | Fit preprocessing and candidates |
| Validation | 13–15 | Select registered candidates; run later diagnostics |
| Frozen test | 16–18 | One final evaluation after audit |

No game crosses a split, and no random row-level split is used. The frozen
test must not be reused for a new selection.

## 9. Frozen selections

| Population | Task | H5 | H10 | H15 |
|---|---|---|---|---|
| All pairs | Regression | OLS | OLS | Ridge, alpha 10 |
| All pairs | Classification | Logistic, C=10 | Logistic, C=10 | Logistic, C=0.1 |
| Nearest defender | Regression | OLS | Ridge, alpha 10 | OLS |
| Nearest defender | Classification | Logistic, C=10 | Logistic, C=10 | Logistic, C=10 |

These 12 selections and their comparators were frozen before one test
execution. Post-audit selection or comparator changes equal zero.

## 10. Representative frozen metrics

Differences are selected minus comparator; negative favors the selected model.

| Population/task | Horizon | Selected | Comparator | Difference | Play-clustered 95% interval |
|---|---:|---:|---:|---:|---|
| All-pair regression MAE | H5 | 1.222541 | 1.299371 | -0.076830 | [-0.090869, -0.063151] |
| All-pair regression MAE | H15 | 3.388624 | 3.920724 | -0.532100 | [-0.666030, -0.404377] |
| Nearest regression MAE | H15 | 1.594830 | 1.602184 | -0.007354 | [-0.100430, 0.077597] |
| All-pair classification log loss | H10 | 0.620705 | 0.659148 | -0.038443 | [-0.048433, -0.028602] |
| Nearest classification log loss | H5 | 0.605635 | 0.667145 | -0.061510 | [-0.077355, -0.048318] |

All 12 primary validation-to-frozen directions agreed, with zero reversals.
Five of six regression intervals were wholly below zero; every reported
classification log-loss and Brier-difference interval was below zero.

## 11. Robustness findings

Coefficient signs were generally stable and removing the separation-only
feature group degraded validation performance most. Defender-context removal
was the next-largest degradation, especially at H10/H15; removing absolute
field location had negligible or occasionally favorable effects.

Performance was not uniform across origin-separation regimes. Comparator
reversals occurred in long-separation validation buckets, and every H15 bucket
had limited support. These are diagnostics, not formal significance claims.

## 12. Calibration findings

On validation, all calibration intercepts were negative, all slopes were below
1, and mean predicted closing probabilities exceeded observed closing rates.
This is consistent with somewhat overconfident raw probabilities. H10 had the
lowest ECE and MCE; H15 uncertainty was widest because its cohort was smaller.
The unmodified probabilities should not be treated as perfectly calibrated or
as calibrated for another season.

## 13. Ethical and interpretive limitations

The designation of a target and the set of supplied defenders reflect
competition packaging, not a complete account of player responsibility.
Model outputs must not be used to assign blame, infer intent, or rank players
without a separately validated design. Predictive association does not imply
causality or football decision quality.

## 14. Data limitations

The source lacks complete 22-player context, all receiving-option
trajectories, direct ball-path information, authoritative coverage labels, and
a second labeled season. Future evaluation covers only entities with supplied
output trajectories. H15 has a substantially smaller, more selective cohort.
Cross-season transport and probability calibration are unknown.

## 15. Release status

- Distribution/version: `gridiron-spatial-intelligence` `0.1.0`
- Milestone 4: **GO**
- Milestone 5: **LIMITED ROBUSTNESS — PROCEED WITH CAUTION**
- Frozen evaluator executions: `1`
- Leakage validation: `PASS`
- Frozen reconciliation mismatches: `0`

This is a validated research baseline and portfolio release, not a
production-ready model.
