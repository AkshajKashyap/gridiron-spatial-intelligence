# Interview Notes

These notes are factual prompts for discussion, not a marketing script.

## 1. Thirty-second explanation

I built a leakage-safe NFL tracking research pipeline around one narrow
question: how geometry at the last supplied input frame relates to later
separation between a competition-designated target and observed defenders. I
audited 5.4 million entity-frame rows, built deterministic cohorts and
reversible coordinates, then evaluated registered linear/logistic baselines
with chronological splits. All 12 primary validation-to-frozen directions
agreed, but robustness analysis found weaker long-separation regimes and
somewhat overconfident probabilities.

## 2. Ninety-second technical explanation

The source initially looked suitable for “passing-window” analysis, but the
audit showed partial entities, incomplete receiving options, and no direct
ball path. I narrowed the claim before modeling. The pipeline assigns every
game to development Weeks 01–12, validation Weeks 13–15, or frozen test Weeks
16–18. Input/output frame IDs are phase-qualified, leftward plays are rotated
into a common attacking direction, and every exclusion reconciles by unit and
reason.

At the immutable last input frame I create target–defender pairs, using all
observed defenders descriptively but only output-valid defenders for H5, H10,
or H15 labels. Features are origin-only geometry. I preregistered constants,
single-feature models, OLS/ridge, and logistic models; preprocessing fits only
on development. After an audit, the frozen split ran once. Selected models
beat registered comparators in the same direction for all 12 primary cells.
Play-cluster bootstraps preserve within-play dependence. Later validation-only
ablations, calibration, and fixed error buckets showed real but nonuniform
signal, so the result is GO with limited robustness rather than a product
claim.

## 3. Central research question

How origin geometry for a competition-designated target and observed
defensive entities relates to future target–defender separation dynamics.

## 4. Why the scope narrowed

The release does not contain complete 22-player tracking, all receiving-option
trajectories, direct ball-path information, or official coverage labels.
Those gaps make complete passing-window, quarterback selection, and
full-field-control claims scientifically weak. Narrowing happened after data
validation and before modeling.

## 5. Most important engineering decisions

- Freeze keys, schemas, cohorts, exclusion ordering, and split boundaries.
- Keep `phase` in every entity-frame key.
- Use an immutable final input frame as origin.
- Normalize direction reversibly while retaining raw coordinates.
- Separate descriptive defenders from future-evaluable defenders.
- Treat H5/H10/H15 as different cohorts rather than filling missing labels.
- Fit preprocessing and candidates only on development weeks.
- Preserve compact aggregate evidence with exact-byte checksums.

## 6. Strongest result

Origin-only multivariable geometry improved registered frozen-test primary
metrics in the same direction as validation for all 12 population–horizon–task
cells, with zero direction reversals. Eight confirmations came from H5/H10, so
the decision did not depend only on the smaller H15 cohort. This is evidence
of reproducible association, not operational impact.

## 7. Most important limitation

The analytic population is conditioned on one supplied target and partial
defensive/output coverage. It cannot answer which receiver a quarterback
should choose or whether the modeled defender had official responsibility.
Only one season is available, so cross-season generalization is unknown.

## 8. Leakage prevention

Games are chronologically blocked. Features come only from the origin or
earlier input frames. Output coordinates, future separation, future
availability, trajectory length, outcomes, IDs, split, and week never enter
the feature matrix. Preprocessing fits on Weeks 01–12. Validation selects
registered candidates; Weeks 16–18 ran once and were not reopened during
Milestone 5.

## 9. Why play-cluster bootstrap?

One play produces multiple defender-pair rows, so row independence would be
false and uncertainty too optimistic. Resampling whole plays retains all
within-play pair rows and aligns the uncertainty unit with the experimental
structure.

## 10. Why H15 is more selective

Every target and defender needs contiguous valid output through frame 15.
Only 8,539 full-season pairs are H15-eligible versus 31,937 at H5. The H15
population therefore represents longer supplied trajectories, not the full
source cohort; every H15 validation error bucket had limited support.

## 11. All-pair versus nearest-defender analyses

All-pair analysis retains each evaluable target–defender relation and answers
a broader local-geometry question. The nearest population keeps one
deterministically closest origin defender per target and changes both sample
weighting and difficulty. “Nearest” is a geometric subset, never an official
coverage label.

## 12. Calibration interpretation

Validation calibration intercepts were negative, slopes below 1, and mean
predicted closing rates above observed rates. The classifiers captured useful
ranking/predictive signal and beat registered log-loss/Brier comparators, but
their raw probabilities were somewhat too extreme and should not be treated
as perfectly calibrated.

## 13. Error-bucket reversals

In several long-origin-separation buckets, the selected model lost to its
comparator even though aggregate performance improved. That indicates
geometric regime heterogeneity, not that the aggregate result is false.
Limited support—especially at H15—means these slices are cautionary
diagnostics rather than subgroup significance claims.

## 14. What comes next without reusing the frozen test

Preregister a new study and reserve a new untouched temporal or cross-season
holdout. Development can explore recalibration, reduced feature sets, and
regime-aware diagnostics using Weeks 01–15 only, but Weeks 16–18 remain final
release evidence and cannot select those changes.

## 15. Likely technical questions

### Why not use a neural network?

The study first needed meaningful, auditable baselines and leakage controls.
The simple models already established signal and exposed heterogeneity. More
capacity is not justified without a new preregistered holdout.

### Why not random cross-validation?

Random rows would mix frames and pairs from the same plays and ignore temporal
deployment order. Week-blocked game splits give a more realistic later-week
test and prevent direct play overlap.

### How is the label defined?

Regression uses future pair separation minus origin pair separation.
Classification is one exactly when that change is negative. Each horizon uses
only its matching future frame.

### Does target designation leak the answer?

It conditions the research population; it is not a feature value and the
project does not predict target selection. The labels are future separation
dynamics, not whether the receiver was targeted.

### How did you verify coordinates?

Leftward plays receive a reversible 180-degree rotation; rightward plays are
identity. Synthetic tests verify round trips, distance preservation, angle
handling, boundary policy, phase keys, and row/index preservation.

### What result was weakest?

Nearest-defender H15 regression improved MAE by only `-0.007354` yards under
the selected-minus-comparator convention, and its 95% play-cluster interval
`[-0.100430, 0.077597]` crossed zero.

### What makes the release inspectable without data?

The repository versions nine compact aggregate JSON files and a deterministic
manifest with roles, sizes, SHA-256 checksums, format/status metadata, and
cross-artifact validation. Raw data, Parquet, predictions, and pair rows stay
excluded.

### Is the model production-ready?

No. The release demonstrates a bounded research result in one competition
season. It has no service layer, operational validation, cross-season
calibration, betting evidence, or production guarantee.
