# Milestone 2 — Target-Centric Geometry and Kinematic Baselines

Milestone 2 obeys
`docs/ANALYTIC_COHORT_AND_COORDINATE_CONTRACT.md`. Analysis is conditioned on
the competition-designated target. Descriptive geometry may use every observed
input defender; future-separation evaluation may use only defenders with valid
supplied output trajectories. Nearest defender is a geometric relationship,
not a coverage assignment.

No quantity is a passing window. This milestone makes no quarterback
target-selection, all-option, full-field defensive-control, or complete
22-player claim. Prediction features may contain no future/output information.
Splits are chronological and game-grouped, never random at row level.

## 1. Analytic cohort builder

**Objective:** Build deterministic, split-tagged indexes for source plays,
target-frames, target-defender pair-frames, prediction origins, and each
evaluable future horizon.

**Inputs:** The 18 audited input/output pairs, supplementary metadata,
`artifacts/milestone_1_validation.json`, and the analytic-cohort contract.

**Outputs:** A reason-coded exclusion ledger; play/entity/frame/origin/horizon
indexes; chronological split assignments; and a machine-readable reconciliation
summary. Derived data remain ignored by Git.

**Focused tests:**

- Reconcile 4,880,579 input rows, 562,936 output rows, 14,108 plays,
  396,914 target-frames, and 46,045 output groups.
- Assert phase-qualified key uniqueness, exactly one target per included
  target-frame, and one split per game/play.
- Account for all 35 zero-defender target-frames without excluding their whole
  plays automatically.
- Assert exact output declarations and report eligibility separately at
  horizons 5, 10, and 15.
- Reject forbidden fields from prediction-feature schemas.

**Quantitative success criteria:** Primary exclusion reasons reconcile exactly
to every source denominator; descriptive geometry retains 396,879
target-frames after the 35 known zero-defender exclusions; every retained unit
has exactly one status and split; and there are zero unexplained join, key,
coordinate, order, or declaration differences from Milestone 1.

**Stop or pivot criteria:** Stop on any unreconciled source total, ambiguous
target, duplicate retained key, split overlap, or feature leakage. If the
horizon-5 future-separation cohort covers fewer than 50% of source plays or
fewer than 30 games in any split, drop future-separation evaluation and retain
target-trajectory plus descriptive geometry only.

**Explicit non-goals:** Coordinate transformation, geometric calculations,
imputation, performance-based filtering, forecasting, or learned modeling.

## 2. Reversible coordinate normalization

**Objective:** Normalize left/right plays to a common positive-x attacking
direction while preserving raw coordinates and every geometric relationship.

**Inputs:** Retained raw entity-frames, audited `play_direction`, raw boundary
flags, and the fixed `L=120.0`, `W=53.3`, `delta=1.0` contract.

**Outputs:** Versioned raw-plus-normalized coordinate fields, normalized
direction/orientation fields where valid, boundary flags, and a transform audit.

**Focused tests:**

- Verify rightward identity and leftward 180-degree rotation.
- Verify round-trip and pairwise-distance preservation within `1e-5`.
- Verify mirrored left/right fixtures normalize identically.
- Verify angle canonicalization and null-angle preservation.
- Verify no clipping and retain/flag the three known Week 3 output rows.

**Quantitative success criteria:** All retained plays have one valid
left/right direction; 100% of transform invariants pass; maximum round-trip
and distance error is at most `1e-5` source coordinate unit; exactly three
known Week 3 rows are retained with flags; and zero coordinates are clipped.

**Stop or pivot criteria:** If any distance/reversibility test fails or play
direction cannot be resolved, stop pooled normalized analysis. Continue only
with raw-coordinate, direction-stratified descriptions until corrected.

**Explicit non-goals:** Fitted scaling, full-field control, separation
calculation, use of unverified angle semantics as features, or visual polish.

## 3. Deterministic geometry calculations

**Objective:** Calculate only the contract-defined target-defender geometry:
pair displacement/distance, nearest and kth-nearest distance, local defender
counts, relative and radial motion, separation change/persistence, and
frames-to-minimum separation.

**Inputs:** Normalized cohort indexes, all observed input defenders for
descriptions, horizon-valid output defenders for future evaluation, and frozen
geometry parameters.

**Outputs:** Target-defender pair-frame and target-frame geometry tables with
evidence level, defender-set definition, units, quality flags, and denominators.

**Focused tests:**

- Hand-check stationary, closing, separating, and tangential synthetic cases.
- Test no-defender, insufficient-k, zero-distance, missing-coordinate,
  nearest-identity-switch, and horizon-edge cases.
- Verify deterministic ties by canonical `nfl_id`.
- Reconcile pair counts to eligible defender counts and verify transform
  invariance.

**Quantitative success criteria:** Every emitted distance is finite and
nonnegative; every count is a nonnegative integer; all hand-computed fixtures
match within `1e-8`; all denominators reconcile to cohort indexes; and the
fixed 12-play audit has zero unexplained geometry discrepancies.

**Stop or pivot criteria:** Stop any quantity whose result depends on an
unresolved transform, timing, role, or defender-set ambiguity. If more than 5%
of otherwise eligible target-frames lose all defenders after required quality
rules, restrict reporting to pair-specific geometry and target trajectories.

**Explicit non-goals:** Coverage assignment, reachability/control maps,
forecasting, learned features, football-value labels, or passing-window claims.

## 4. Descriptive full-release analysis

**Objective:** Describe target-centric separation and closing behavior across
the audited release and test whether any threshold/persistence convention is
stable enough to freeze.

**Inputs:** Frozen cohort/split indexes, deterministic geometry, exclusion
ledger, and predeclared training-only sensitivity grids.

**Outputs:** Week/split/role/horizon distributions, defender-coverage
denominators, continuous-geometry summaries, threshold/persistence sensitivity
tables, and a limitation report.

**Focused tests:**

- Reconcile every summary to its row, play, and game denominator.
- Verify deterministic bins and game-level aggregation.
- Assert that definitions are not changed from frozen-test values.
- Repeat summaries across predeclared threshold/persistence settings.

**Quantitative success criteria:** All summary denominators reconcile exactly;
100% of reported values name their unit, split, and evidence level; and a
thresholded label is frozen only if training and validation prevalence both
remain between 5% and 95% and the substantive conclusion is stable across the
predeclared sensitivity set.

**Stop or pivot criteria:** If prevalence or episode conclusions are unstable,
do not create a thresholded label or episode claim. Continue with continuous
distance and motion descriptions only.

**Explicit non-goals:** Test-driven threshold selection, player rankings,
causal interpretation, model training, all-receiver opportunity, or QB
analysis.

## 5. Constant-position and constant-velocity baselines

**Objective:** Establish transparent frame-based trajectory and
future-separation baselines at horizons 5, 10, and 15.

**Inputs:** Eligible two-frame histories, normalized origin coordinates,
horizon-specific supplied output labels, and a fixed output-evaluable defender
set shared by truth and both baselines.

**Outputs:** Constant-position and one-step finite-difference
constant-velocity forecasts, trajectory errors, future-separation errors, and
forecast quality flags.

**Focused tests:**

- Match exact stationary and linear synthetic trajectories.
- Assert every feature source is input phase at or before the origin.
- Verify horizon masking, identical defender sets, and no coordinate clipping.
- Audit the feature matrix for output, landing, outcome, and declaration
  fields.

**Quantitative success criteria:** Forecasts are finite for 100% of eligible
samples; all horizon denominators reconcile; synthetic errors are zero within
`1e-8`; and constant velocity has game-level validation MAE no worse than
constant position at the primary 5-frame horizon.

**Stop or pivot criteria:** If constant velocity is worse than constant
position at every usable horizon, materially violates the declared boundary
diagnostics, or requires future information, stop predictive escalation and
publish descriptive/constant-position results.

**Explicit non-goals:** Learned models, smoothing with future frames, physical
ball flight, probabilities, seconds-based claims without authoritative cadence,
or neural networks.

## 6. Evaluation and error slicing

**Objective:** Evaluate frozen baselines under chronological, game-grouped
splits and explain their errors without tuning on the test period.

**Inputs:** Frozen baseline predictions; weeks 1–12 train, 13–15 validation,
and 16–18 test assignments; cohort flags; and game-level metadata available at
the origin.

**Outputs:** ADE/FDE, x/y error, separation MAE/RMSE, game-block confidence
intervals, and slices by horizon, week, role/position, initial separation,
defender count, and boundary flag.

**Focused tests:**

- Assert zero game/play overlap across splits.
- Match metrics on hand-calculated fixtures.
- Verify deterministic game-block resampling and exact slice denominators.
- Assert train-only fit provenance and a single frozen-test evaluation path.

**Quantitative success criteria:** Split totals reconcile to 180/44/48 games
and 3,236,116/795,547/848,916 input rows; every metric includes its denominator
and 95% game-level interval; bootstrap results reproduce under a fixed seed;
and the frozen test is evaluated exactly once after validation choices freeze.

**Stop or pivot criteria:** Any leakage or split overlap invalidates the
evaluation. If an apparent pooled gain reverses across validation weeks,
primary horizons, or game-level intervals, report no robust baseline gain and
stop predictive escalation.

**Explicit non-goals:** Random row-level splitting, test-set tuning, causal
player/team claims, cross-season generalization, learned-model promotion, or
leaderboard optimization.

## 7. Limited validation visualizations

**Objective:** Verify that normalized geometry and baseline outputs remain
consistent with the fixed representative-play evidence.

**Inputs:** The 12 deterministic Milestone 1 plays, normalized coordinates,
derived geometry, origins, observed futures, baseline forecasts, split labels,
and quality flags.

**Outputs:** Exactly one reproducible static panel or lightweight animation for
each of the 12 fixed plays, clearly separating observed history, origin,
observed future, and forecast.

**Focused tests:**

- Verify play/split identifiers and raw-to-normalized correspondence.
- Show only past trails before the origin.
- Match displayed distances, defender IDs, horizons, and forecasts to derived
  tables.
- Require visible boundary and observed/derived/predicted labels.

**Quantitative success criteria:** Exactly 12 artifacts are generated; all 12
match their source tables; all future layers are visually distinct from input
features; and manual review records zero unexplained coordinate, timing,
identity, or leakage discrepancy.

**Stop or pivot criteria:** Any unexplained visual/table disagreement blocks
reporting and returns work to the earliest failing cohort, transform, geometry,
or baseline task. Plots may not be cosmetically patched around bad data.

**Explicit non-goals:** Dashboard or UI development, hand-picked highlights,
coverage diagrams, full-field control, ball paths, QB decisions, deployment,
or passing-window visualization.

