# Gridiron Spatial Intelligence — Project Plan

## 1. Objective

Build an evidence-led spatiotemporal analysis of how receiver opportunities
develop during NFL passing plays.  The project studies the chain

`player movement -> receiver/defender interaction -> local space -> candidate geometric opportunity`

before considering quarterback decisions.  Its contribution is not a generic
play-outcome predictor: it is a reproducible way to describe, forecast, and
visualize the geometry around potential receivers.

The first deliverable should make modest, testable claims. It will measure
whether a receiver is geometrically separated from nearby defenders now and in
the near future. It will **not** call that separation a passing window, claim
that it is a throw the QB should make, or estimate completion probability.

### Decision gates

Do not commit to a learned model until these gates pass:

1. A local dataset audit establishes the exact schema, frame rate, time range,
   and licenses/competition rules.
2. Coordinates can be normalized and animated without discontinuities.
3. Geometric labels are stable under a small, predeclared range of thresholds.
4. A leakage-safe constant-velocity baseline is established.
5. A candidate learned model is promoted only if it improves held-out,
   time-based evaluation and is calibrated when it returns probabilities.

## 2. Precise initial research question

**At a pre-throw frame `t`, can the recent motion and contemporaneous spatial
configuration predict whether an eligible receiver will enter and sustain a
thresholded geometric-separation state during the next `h` seconds?**

The primary horizon set should be `H = {0.5, 1.0, 1.5}` seconds, subject to
the amount of tracking remaining before the ball is released.  The primary
analysis population is pass-play frames from the snap (or the first valid
tracking frame) through, but excluding, the release frame.  A receiver must be
defined from roster position and validated play participation; this is a
dataset-audit item, not an assumption that every offensive player is a route
runner.

There are three deliberately separate terms. They must not be collapsed in
reporting or visualization:

- **Separation state:** a deterministic label derived from observed locations
  and a declared threshold/persistence rule. This is the MVP target.
- **Candidate geometric opportunity:** an interpretive summary of separation,
  relative motion, and perhaps a declared reachable-space model. It is still
  not a pass window.
- **Passing window / pass usability:** a conditional construct at a proposed
  arrival point and time, incorporating a feasible ball path, receiver catch
  reach, defender contest reach, and QB/ball timing. It is an optional research
  extension only if the source actually supports those inputs.

An observed target receiver, a completed pass, or a landing location may be
used for stratified analysis after model evaluation. They are not inputs to the
receiver separation-state model. Otherwise the task becomes a target prediction
or a post-throw reconstruction problem.

## 3. Data requirements and current evidence

### Repository inspection

The repository is a clean research skeleton.  It has a Python package stub,
data/artifact directories protected by `.gitignore`, `docs`, `notebooks`, and
`tests`; it contains no tracking files, schemas, or data documentation.  The
initial work should therefore add documentation and an audit notebook/report,
not a data pipeline or model framework.

### Candidate: NFL Big Data Bowl 2026

The 2026 Big Data Bowl is the preferred initial candidate because its stated
task is player-movement prediction during the ball-in-air phase.  NFL Football
Operations says the released inputs concern movement before the ball is thrown,
and that the competition evaluates predictions against 2025 Weeks 14–18 while
using 2023–24 seasons for the earlier data. [NFL Football Operations — Big
Data Bowl](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl)
The Kaggle overview further says the pre-throw data include NGS tracking and
the targeted offensive player and landing location. [Kaggle — NFL Big Data
Bowl 2026 Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview/abstract)

This makes it promising for receiver/defender geometry immediately before a
throw and for a later ball-arrival extension.  It does **not** by itself prove
that all post-snap route development, ball trajectory, coverage assignment, or
pass outcome fields are available in every split.  The competition's intended
prediction target also differs from this project's initial target, so no
competition metric should be adopted uncritically.

### Confirmed, unknown, and required before modeling

| Item | Status | Planning implication |
| --- | --- | --- |
| Local tracking files or schema | **Not present** | Nothing in this repository confirms columns, records, or data quality. Download/access and record a manifest before implementation. |
| Player tracking before release | **Publicly described** | Enough in principle for pre-throw geometry and motion history. Confirm exact temporal coverage per play. |
| Competition focus / post-throw outcomes | **Publicly described** | Predictions are evaluated on movement while the ball is in the air; do not assume those future coordinates are present in an unrestricted local training file. |
| Targeted receiver and pass landing location | **Publicly described** | Useful only for post-hoc analysis or a separately declared, conditional task. Validate field names and split availability. |
| Seasons / held-out organization | **Publicly described at a high level** | Treat the stated 2023–24 versus 2025 Weeks 14–18 arrangement as a useful chronology, but inspect the actual train/test files and labels. |
| Sampling frequency | **Not locally verified** | NFL material describes Next Gen Stats tracking at 10 Hz, but the acquired competition extract must be checked rather than hard-coded. |
| Coordinates, speed, acceleration, direction, orientation | **Expected from common NGS-style data; unconfirmed here** | Require a data dictionary and check units, coordinate origin, direction convention, missingness, and whether values are measured or derived. |
| Ball coordinates | **Unknown locally** | Needed to align release timing and later path-based work; not required for the first separation label. |
| Event timestamps / release event | **Unknown locally** | Need an authoritative release-frame definition. A `frameId` is not automatically a wall-clock timestamp. |
| Pass outcome fields | **Unknown locally** | Do not promise completion or target analysis until audited. |
| Explicit individual coverage assignments | **Unknown / should not be presumed** | Initial defender interactions use proximity and reachable-space rules, not asserted man-coverage labels. |
| Player identity, team, position, game/play IDs | **Required** | Necessary to join plays and players, select pass plays, define sides, and create leakage-safe groups. |

### Dataset acceptance checklist

Before stage 1 is considered complete, save `artifacts/data_audit.json` and a
short audit notebook with: file hashes and row counts; column names/dtypes;
keys and duplicate checks; cadence distribution; coordinate extrema; event
vocabulary; missingness; player/team/position joins; pass-play counts; number
of valid pre-release frames; labels available by split; and a license/terms
note.  The raw source files remain in `data/raw/` and are never committed.

If the 2026 extract lacks enough pre-release frames or a reliable release
marker, fall back to a public pass-oriented BDB release with a documented
schema.  Preserve the same research question, report the dataset change, and
do not blend releases without a documented harmonization study.

## 4. Proposed play representation

### Coordinate frame and sample

Normalize every play so the offense attacks in the positive `x` direction.
Apply the same geometric transform to location, velocity vector, heading, ball
location, and derived features; never transform `x` alone.  Retain raw values
alongside normalized values for auditability.  Use field-relative coordinates
for plotting and optionally ball-relative coordinates for model inputs.  The
normalizer must be verified with before/after animations from plays in both
directions.

At time `t`, a receiver-centric sample contains:

- `k` prior frames of each participating offensive player, defender, and ball
  (when available), plus masks for absent data;
- normalized position `(x, y)`, velocity `(vx, vy)`, optionally acceleration,
  speed, and orientation after their definitions have been verified;
- roles: offense/defense/ball, positional group, receiver-candidate flag, and
  a stable player identity only for splitting/reporting—not as an unrestricted
  shortcut feature;
- play context that is genuinely known by `t` (for example down, distance, or
  formation only after a missingness and leakage review);
- pairwise receiver-defender geometry and a fixed-size nearest-defender view.

The natural full-play representation is a time-varying graph: players (and
optionally the ball) are nodes, with node motion histories and edges defined by
distance, relative bearing, or an interpretable reachability rule.  The MVP
does not need a graph neural network; a receiver-centered tabular view is a
lossless enough first projection for a small number of nearest defenders.

### Measurement and label contract

The plan uses three evidence levels. Keep the level next to every quantity in
tables, figures, and saved data:

1. **Observed:** a source field (for example, location, time index, team) after
   its schema and units have been audited.
2. **Deterministically derived:** a reproducible transformation of observed
   fields with declared parameters (for example, normalized coordinates,
   Euclidean separation, a thresholded separation episode, or a finite
   difference). It is not ground truth about football intent.
3. **Model-dependent:** a forecast, probability, reachability map, or score.
   Its validity comes from held-out evaluation and its assumptions, never from
   its visual plausibility.

For receiver `r`, defender `d`, and normalized positions `p`:

`d_min(r,t) = min_d ||p_r(t) - p_d(t)||`

Given a pre-registered threshold `tau` and persistence window `q`, define the
**thresholded separation state** at a frame as
`S(r,t) = 1[d_min(r,t) >= tau]`. Define the future-separated target at horizon
`h` as:

`Y_h(r,t) = 1[S(r,u)=1 for enough frames in [t+h, t+h+q]]`.

Use at least two sensible `tau` values in sensitivity analysis; choose one
primary value before comparing models. A threshold is a modeling convention,
not an objective football truth or a passing-window label. Frames without an
observed future interval are excluded for that horizon rather than silently
labeled negative.

### Progressive quantities

| Quantity | Evidence level | Definition | Assumption / caveat |
| --- | --- | --- | --- |
| Nearest-defender separation | Deterministically derived | `d_min`, pairwise Euclidean distance. | The nearest defender need not be responsible for the receiver. |
| Relative radial velocity | Deterministically derived | Change in receiver–defender distance per second. | Noisy tracking and changing nearest defender can create jumps. Report pair-specific and nearest versions. |
| Closing time | Deterministically derived | `d / max(-d_dot, eps)` when a defender is closing. | A kinematic extrapolation; cap it and do not read it as a true time-to-cover. |
| Separation growth/decay | Deterministically derived label | Future change in `d_min`. | A valid evaluation outcome, but never an input to a forecast at `t`. |
| Local defender density | Deterministically derived | Sum of distance-decay kernels around a receiver. | Kernel bandwidth is a modeling choice; it ignores assignment and path obstruction. |
| Leverage | Deterministically derived | Defender position/velocity projected on a declared direction. | The reference direction is an assumption; do not call this coverage quality by default. |
| Reachable / controlled area | Model-dependent | Grid points each player can reach first under a declared motion model. | Reaction time, speed cap, turn cost, and tie rules are assumptions. |
| Future-separated probability | Model-dependent | `P(Y_h=1 | information at t)`. | Probability under this dataset and threshold, not catch probability. |
| Separation episode | Deterministically derived | Maximal consecutive interval satisfying `S`. | It is not a passing window without a ball-arrival/contest definition. |

## 5. Baselines and modeling progression

The progression earns complexity: each stage has a narrow claim, a naive
comparator, and a stop/go criterion.  All metrics below are reported by
horizon, position group, route-time bucket, and held-out game—not only as a
single pooled number.

| Stage | Question / inputs | Output and baseline | Evaluation and meaningful success | Depends on |
| --- | --- | --- | --- | --- |
| 0. Data contract | Can the source be joined and interpreted safely? IDs, schema, frames, events. | Retained analytic table and audit report; baseline is raw-source counts. | Zero unresolved key, duplicate, or order errors in retained rows; every exclusion and timing ambiguity is counted and explained. | Dataset access |
| 1. Geometry | What does separation look like at a frame? Positions and roles. | `d_min`, nearest-defender identity, density; baseline is direct distance calculation. | Coordinate invariants pass; 12 stratified manual play checks contain no unexplained direction, timing, or entity errors. | 0 |
| 2. Exploratory animation | Do transforms and derived quantities agree with the play? Normalized tracks and stage-1 features. | Reproducible static panels/animations. | Same 12-play audit plus a random held-out sample; no claim of predictive skill. | 1 |
| 3. Relative motion | Does current motion describe future separation? Short history of receiver/defender tracks. | Future `d_min` change and closing-time summaries; baseline is persistence. | Report MAE by horizon with game-block bootstrap intervals; this is descriptive, not a model-promotion gate. | 1 |
| 4. Kinematic forecast | Can simple physics forecast receiver and defender positions? Current position and velocity. | Future location and separation; persistence and constant velocity are the models. | ADE/FDE and separation MAE by horizon. Retain constant velocity only if its primary-horizon game-level MAE is no worse than persistence; otherwise diagnose cadence/heading before proceeding. | 3 |
| 5. Learned separation / state | Does learned context improve a simple forecast? Engineered history, geometry, context. | Regression for future separation and/or calibrated `P(Y_h=1)`; baselines are persistence and constant velocity. | At the primary horizon, 95% game-bootstrap CI must favor the best baseline and show at least 0.10 yd **and** 3% lower separation MAE. For a state classifier, require PR-AUC +0.02, Brier −0.01, calibration slope 0.8–1.2, and ECE <=0.03 on the locked test. | 4 |
| 6. Spatial control | Does a reachability model add beyond nearest distance? All player tracks and a declared reach model. | Receiver-accessible area, defender control map; baseline is Voronoi/nearest-player control. | Sensitivity analysis over every motion assumption and incremental held-out value beyond `d_min`; abandon the map if its conclusion reverses across reasonable assumptions. | 1, 4 |
| 7. Geometric-opportunity score | Can transparent components add beyond distance? Separation, closing, density, control. | A component report and clearly named score; baseline is `d_min` alone. | Same locked-test improvement/calibration criteria as stage 5 plus ablations. This is explicitly not a passing-window score. | 5 or 6 |
| 8. Player analysis | Are patterns stable and useful across people? Held-out per-frame/episode estimates. | Player/position summaries with uncertainty; baseline is league-position average. | At least 200 eligible frames and 20 separation episodes per displayed player; shrinkage/intervals and threshold robustness required. | 7 |
| 9. Optional QB analysis | Conditional on a valid spatial model, how do actual targets relate to opportunities? Add QB/ball/target labels only for a separate task. | Descriptive target alignment or explicitly labeled choice model. | Held-out conditional evaluation; never equate correlation with decision quality. | 7, verified labels |

### Model families: use only when their extra representation is justified

| Family | Consumes | Strength / added complexity | Role in this project |
| --- | --- | --- | --- |
| Persistence and constant velocity | Current state, optionally one velocity vector. | Transparent physics floor; cannot handle cuts, interactions, or uncertainty. | Mandatory baselines for all future-state tasks. |
| Regularized linear / logistic model | Engineered history features, pairwise geometry, context. | Auditable coefficients and low data demand; linear effects may miss route changes. | First learned model. |
| Tree ensemble (e.g., gradient boosting) | Same fixed-width features. | Captures nonlinear thresholds/interactions; needs careful temporal grouping and calibration. | Strong tabular benchmark before deep learning. |
| GRU/LSTM | Ordered receiver/pair histories, masks. | Learns compact temporal state; adds sequence length, padding, and weaker direct interpretability. | Consider only if sequence information beats engineered histories. |
| Temporal convolution | Fixed-duration multivariate histories. | Efficient local-motion patterns; receptive field choices matter. | A practical sequence alternative to recurrent models. |
| Transformer | Full multi-agent history with role/position embeddings and masks. | Flexible long-range interactions; data hungry and easily overfits player/team identity. | Later comparison, not a default. |
| Spatial / temporal GNN | Dynamic player graph with node motion and interaction edges. | Matches multi-agent structure and can model changing matchups; edge definition, pooling, and evaluation are substantially harder. | Research extension after control maps and tabular baselines are strong. |
| Graph + temporal architecture | Sequence of dynamic graphs. | Most expressive but hardest to debug, attribute, and benchmark. | Only for a clearly stated hypothesis that simpler models fail to meet. |

For learned location prediction, predict a distribution or quantiles where
possible, not just a point. For a classifier, calibrate on validation data
only. A more complex model that improves a pooled score but degrades
calibration, later horizons, or important position groups is not a clear win.
Deep sequence or graph models are prohibited until the stage-5 linear/tree
benchmark clears its locked-test promotion criteria and a documented error
analysis identifies a specific interaction or history limitation that the added
representation could address.

## 6. Leakage-safe methodology and evaluation

### Information boundary

For a sample indexed by receiver and frame `t`, features may use only data at
or before `t` (and declared pre-snap context).  Its label may use `t+h` onward.
Implement this in the feature API: pass an explicit cutoff frame and assert
that every source frame is no later than the cutoff.  Store feature cutoff,
label interval, and data split with each example.

Do not use:

- tracking frames after the cutoff, including smoothed values whose filter
  looks ahead;
- release/arrival/outcome events, target receiver IDs, catch location, or pass
  outcome as features in the initial task;
- labels or aggregates calculated over the entire play, game, or season;
- a defender assignment inferred using future motion;
- player/team outcome rates computed with validation/test rows;
- random frame-level splits, which place adjacent frames and the same play in
  train and test.

Even roster position can be problematic if it is taken from a post-season
table rather than the game-day source.  Document feature availability and
provenance in the data contract.

### Splits

Use a final chronological holdout containing later games/seasons, never tuning
against it. All frames of a play, and all plays in a game, belong to one split.
For the candidate source, the intended default—subject to the M1 audit
confirming labels—is train on 2023, validate on 2024, and test once on the
available labeled 2025 Weeks 14–18 outcome period. Within 2023, use expanding,
game-grouped time folds for development; do not shuffle frames or plays. If the
2025 labels are unavailable under the acquisition terms, reserve the last
chronological block of 2024 games as the final test and state that no
competition-test claim is being made. `GroupKFold` without chronology is only
a sensitivity analysis, never the primary estimate.

Players and teams naturally recur. Chronology is primary because it reflects
deployment and avoids future season knowledge. Additionally report a
team-held-out or player-heavy holdout only as a harder robustness diagnostic
where sample size permits. It is not a replacement for the chronological test:
players and teams recurring across future games is the natural deployment
setting.

Fit every learned transform—imputation, scaling, threshold calibration,
feature selection, hyperparameter choice, and probability calibration—inside
the training portion of each fold.  Bootstrap confidence intervals by **game**
or block-bootstrap by play, not by nearly identical frames.

### Metrics and naive comparators

| Task | Main metrics | Required comparators / diagnostics |
| --- | --- | --- |
| Player trajectory prediction | ADE by horizon, final displacement error (FDE), x/y error, negative log likelihood or interval coverage for probabilistic forecasts. | Persistence and constant velocity; physical plausibility (speed/acceleration violations); errors by player role and motion phase. |
| Future separation regression | MAE and RMSE of `d_min(t+h)` and change in separation; rank correlation where ranking is useful. | Persistence (`d_min(t)`) and separation computed from constant-velocity positions; error by horizon and open/closed state. |
| Becomes-separated classification | PR-AUC (primary if positive states are rare), ROC-AUC, Brier score, log loss. | Majority/persistence classifier; threshold confusion matrix; prevalence and position-stratified results. |
| Probability quality | Calibration curve, calibration intercept/slope, expected calibration error reported with bin choice, Brier decomposition where feasible. | Raw versus validation-calibrated probabilities; reliability by horizon. |
| Separation episodes | Frame-level precision/recall/F1 only as secondary; onset/close timing MAE, episode-duration MAE, event precision/recall within a tolerance, and interval IoU. | Current-state/persistence episode baseline; report false starts and missed episodes separately. |
| Spatial-control maps | Agreement with a declared simulated/reachable baseline; association with future separation state while controlling for current separation. | Static Voronoi/nearest-player map; sensitivity to speed/turn/reaction assumptions. |

Never report only accuracy: a class-imbalanced separation-state label can make a model
look strong by mostly predicting closed.  For every headline comparison,
publish the horizon, label threshold/persistence rule, denominator, confidence
interval, and exact split.

## 7. Visualization plan

Do not build a UI in the MVP implementation phase.  First create reproducible
notebook figures and exported HTML/MP4/GIF artifacts.  The minimum compelling
visualization is a single-play explorer view with a frame slider or animation:

- a regulation field in the normalized coordinate frame, labeled with the
  direction of attack;
- offense, defense, and ball with stable color/marker conventions and player
  labels; observed trajectory trails limited to past frames;
- selected receiver's nearest-defender link, current separation, and relative
  closing/separation-state indicator;
- an explicit cursor for the prediction cutoff and a dashed future forecast,
  visually distinct from observed future positions;
- a small time series of observed separation and thresholded separation
  episodes; add a predicted state probability only in the later learned-model
  stage; and
- toggles for interaction links, velocity arrows, predicted locations, and
  later, spatial-control contours.

The explorer must state the play ID, split, horizon, model version, label
definition, and whether a layer is observed, derived, or predicted.  It should
default to a fixed small gallery selected before seeing model results plus a
random held-out sample, avoiding only highlight-reel examples.  Spatial-control
regions belong only after their reachability assumptions are shown in a legend.

## 8. Scope: MVP, strong v1, and extensions

### MVP

A portfolio-worthy minimum is:

1. Audit and normalize one verified pass-tracking dataset.
2. Define and sensitivity-test an interpretable nearest-defender separation
   episode with a predeclared horizon and persistence rule.
3. Build constant-position and constant-velocity forecasts for future
   separation.
4. Evaluate those baselines with leakage-safe, time-based splits and
   game-block uncertainty intervals.
5. Publish a concise report with failure examples and an
   animated/static play gallery that makes the label and model behavior legible.

This is enough if it is reproducible, honest about the difference between
separation and availability, and produces a clear held-out baseline result. It
does not require a learned model, deep learning, inferred coverage assignments,
a web app, or QB grades.

### Strong v1

Add one regularized or tree-based multi-horizon separation/state model, with
calibration and ablations against distance alone; then add a validated
reachability/control map, receiver/position summaries with shrinkage and
uncertainty, and a polished shareable play explorer. A temporal model is
justified only if it materially improves the locked strong-v1 tabular benchmark.

### Optional research extensions

- Path-aware window feasibility conditioned on a proposed pass arrival point
  and an explicitly modeled ball-flight assumption.
- Latent or probabilistic receiver-defender responsibility modeling, clearly
  separated from ground-truth coverage labels.
- Graph-temporal models of all 22 players.
- A target-selection study conditioned on eligible receivers and stated
  opportunities.
- QB decision analysis only with an outcome/throw model, counterfactual care,
  and language limited to association unless causal assumptions are defended.

## 9. Failure modes and mitigations

| Risk | Why it weakens the project | Mitigation |
| --- | --- | --- |
| Insufficient or mismatched data | The candidate may truncate the relevant pre-throw period or withhold labels. | Run the acceptance checklist before feature work; pivot datasets transparently if needed. |
| “Open” is ambiguous | A distance threshold can be geometrically true yet football-irrelevant. | Name it a thresholded separation state, show threshold sensitivity, and defer football-value claims. |
| Separation is mistaken for pass availability | Ball path, QB timing, catch radius, leverage, and intervening defenders matter. | Maintain the two-target distinction; add path-aware modeling only with the required observations/assumptions. |
| Orientation/leverage are ignored or misread | Proximity alone misses where defenders are moving or facing. | Add relative velocity first; audit orientation semantics before using it; call leverage a defined projection. |
| Coverage is invented from proximity | The closest defender may not have responsibility. | Use “interaction” rather than “assignment” until labels or a separately evaluated latent model exist. |
| Temporal or football-specific leakage | Future frames, release/outcome metadata, and random frame splits inflate results. | Enforce cutoff-aware features, group by play/game, use chronological holdout, and audit provenance. |
| Dataset selection bias | Only throws, particular game periods, or competition-selected plays may not represent all opportunities. | Describe the sample universe and avoid claims about unthrown opportunities or the full NFL. |
| Complex model without baseline gain | Deep architecture complexity can hide no real improvement. | Require baseline improvements, uncertainty intervals, ablations, and calibration before promotion. |
| Visualization hides weak empirical results | Attractive examples can be cherry-picked. | Show random held-out plays, failures, fixed galleries, and model metadata alongside metrics. |
| Identity shortcuts / small player samples | Player or team IDs can memorize tendencies; player rankings become noisy. | Exclude identity from default predictive features; use group/chronological splits and partial pooling or minimum samples. |

### Stop, pivot, and promotion criteria

The project must change direction when evidence warrants it; a polished
visualization is not a reason to continue. Apply these rules after M1 and again
after the locked baseline evaluation:

| Trigger | Required decision |
| --- | --- |
| Cannot reconstruct a temporally ordered pre-release sequence with offense/defense identity and usable coordinates for a substantial retained pass-play sample | **NO GO** for this source and research question; acquire a documented alternative rather than impute the missing structure. |
| Coordinates and player locations are usable, but no reliable release marker, player positions, or target/arrival metadata exists | **LIMITED GO**: restrict the study to unconditioned within-play separation dynamics; remove pre-throw timing, receiver eligibility, or pass-usability claims as applicable. |
| Thresholded-separation prevalence or episodes are unstable across the predeclared threshold/persistence sensitivity set | Do not fit a classifier or create player rankings. Keep only descriptive geometry or redesign the label before proceeding. |
| Constant-velocity and persistence are indistinguishable or poor at every usable horizon | Retain descriptive separation analysis; do not imply motion predictability. Diagnose frame cadence/coordinate quality before considering a learned model. |
| A learned/tabular model fails the stage-5 locked-test promotion thresholds | Stop model escalation. Publish the negative baseline result or pivot to visualization/measurement, not neural networks. |
| A control-map conclusion changes under reasonable motion assumptions | Do not report a control score; keep its assumptions as an exploratory appendix or abandon the extension. |

## 10. Milestone roadmap

| Milestone | Deliverable | Completion criterion |
| --- | --- | --- |
| M1 — Data and spatial sanity | Manifest, schema note, validation report, and minimal static/animated checks. | `docs/MILESTONE_1.md` exits GO, LIMITED GO, or NO GO. |
| M2 — Geometry and label study | Coordinate contract, separation definitions, prevalence/episode sensitivity report. | Primary separation-state label is frozen before model comparison. |
| M3 — Kinematic baselines | Persistence and constant-velocity evaluation report. | Held-out trajectory/separation metrics with game-level intervals and the stage-4 gate applied. |
| M4 — MVP report | Reproducible narrative, baseline play gallery, artifact index. | A reader can reproduce the baseline result and inspect representative behavior. |
| M5 — Strong-v1 tabular model | Linear/tree model, calibration, ablation, failure analysis. | Stage-5 promotion criteria pass or the model path stops. |
| M6 — Advanced-method decision | Control-map or temporal-model proposal. | Added representation addresses a measured strong-v1 limitation and has a locked benchmark. |

## 11. Minimal repository architecture

Keep the current layout.  Add code modules only when a milestone needs them:

```text
docs/
  PROJECT_PLAN.md                 # this plan
  MILESTONE_1.md                  # data-only acquisition and sanity-check plan
  data_audit.md                   # created at M1
notebooks/
  01_data_audit.ipynb
  02_geometry_and_labels.ipynb
  03_mvp_evaluation.ipynb
src/gridiron_spatial/
  __init__.py
  io.py                            # introduced at M1: explicit loaders/schema checks
  geometry.py                      # introduced at M2: transforms and measurable geometry
  labels.py                        # introduced at M2: cutoff-safe target construction
  evaluation.py                    # introduced at M3: grouped splits and metrics
  visualization.py                 # introduced at M5: reusable plotting primitives
tests/
  test_geometry.py
  test_labels.py
data/raw/                          # source data, ignored by git
data/processed/                    # derived, reproducible tables, ignored by git
artifacts/                         # figures, reports, manifests, ignored by git
```

Avoid a database, orchestrator, feature store, service layer, or model registry
until the research has a proven need.  Keep configuration close to the
notebook/experiment until there are repeated runs worth standardizing.  Raw and
derived data must remain outside version control; commit only small manifests,
schema descriptions, code, tests, and report artifacts that do not violate the
source terms.

## 12. Immediate next action

Obtain approval to execute Milestone 1, acquire the authorized dataset into
`data/raw/`, and perform only the loader/audit and minimal coordinate/timing
checks described in `docs/MILESTONE_1.md`. Revisit this plan after the audit
replaces the currently unknown schema items with confirmed facts.
