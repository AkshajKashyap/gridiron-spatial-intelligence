# Analytic Cohort and Coordinate Contract

Status: binding Milestone 2 specification. No cohort, geometry, or model has
been implemented by this document.

## 1. Scope and claim boundary

Milestone 1 ended in a **LIMITED GO**. This contract therefore conditions every
analysis on the competition-designated targeted receiver and the defensive
entities actually supplied by the NFL Big Data Bowl 2026 Analytics release.
The research question is:

> For a competition-designated target and the observed coverage defenders, how
> do separation and closing behavior evolve from pre-release geometry into the
> supplied post-release trajectory?

The data support target-centric receiver/defender geometry, separation and
closing behavior, trajectory prediction, and scoped comparison of the supplied
input and output sequences. They do not support claims about all receiving
options, full-field defensive control, quarterback target selection, complete
22-player tactics, or direct ball-path obstruction. No quantity in this
contract is a passing-window metric. That term is reserved for a later source
or extension with sufficient full-field and ball-path context.

The audited source universe is 272 games, 14,108 tracked plays, 4,880,579 input
entity-frame rows, 562,936 output entity-frame rows, and 46,045 matched output
player sequences. All 14,108 plays have exactly one targeted receiver.

## 2. Units of analysis

Input and output `frame_id` values occupy separate sequences and both start at
1. A phase field is therefore mandatory in every derived key.

| Unit | Binding key and definition | Use |
|---|---|---|
| Play | `(game_id, play_id)` | Cohort inclusion, split assignment, and play-level summaries. A play is never treated as independent of its own frames or entities. |
| Tracked entity | `(game_id, play_id, nfl_id)` plus audited role/side | One supplied player trajectory within one play. The same `nfl_id` in another play is a repeated player, not the same statistical sample. |
| Entity-frame | `(game_id, play_id, phase, frame_id, nfl_id)`, where `phase in {"input", "output"}` | Smallest observed location unit. This is the candidate primary key for derived tracking tables. |
| Target-frame | `(game_id, play_id, phase, frame_id, target_nfl_id)` | Descriptive target-centric geometry at one frame. It exists only when exactly one target row with valid coordinates is present. |
| Target-defender pair-frame | Target-frame key plus `defender_nfl_id` | Pair-specific distance and relative-motion quantities. One target-frame produces one row for every eligible observed defender, without implying coverage responsibility. |
| Prediction origin | `(game_id, play_id, target_nfl_id, origin_kind, origin_frame)` | The information cutoff. The primary cross-boundary origin is the maximum raw integral input frame of the play before cohort exclusions, represented as relative frame 0; required entities must be valid at that immutable boundary. |
| Future horizon | A positive integer `h` measured in output frames after the primary origin | A label/prediction index, never an independent sample. Primary horizons are 5, 10, and 15 frames, subject to label availability. |
| Entity-origin-horizon | Prediction origin plus predicted `nfl_id` and `h` | Trajectory-prediction evaluation unit. |
| Target-origin-horizon | Prediction origin plus `h` and a declared evaluable defender set | Future-separation evaluation unit. |

Descriptive geometry is computed at target-frame and pair-frame level.
Trajectory prediction is evaluated at entity-origin-horizon level. Future
separation is evaluated at target-origin-horizon level. Metrics may be
calculated per row, but uncertainty and headline comparisons are aggregated
first by play and then by game.

Player-level summaries are post-evaluation aggregates, not training samples.
They must first collapse repeated frames within each play and report both play
and frame denominators. Player rankings are postponed until a separation
episode is frozen; any later displayed player requires at least 200 eligible
frames and 20 separation episodes, as required by the project plan. Repeated
players and teams across weeks remain allowed because that reflects the
intended chronological setting; identity is not a default predictive feature.

## 3. Analytic-cohort rules

### 3.1 Named cohorts

The implementation must materialize and count these distinct cohorts:

1. **Source play cohort:** all 14,108 audited `(game_id, play_id)` values.
2. **Descriptive target-frame cohort:** input target-frames with a valid target
   and at least one valid observed defensive entity at that frame.
3. **Primary-origin cohort:** plays whose target has at least 2 consecutive
   valid input frames ending at the fixed last supplied input frame.
4. **Trajectory cohort at `h`:** prediction candidates in the primary-origin
   cohort with valid, contiguous output frames `1..h`.
5. **Future-separation cohort at `h`:** target origins for which the target and
   at least one eligible defender both have valid output coordinates through
   frame `h`.

For descriptive input geometry, the defender set is every same-frame entity
with `player_side == "Defense"` and `player_role == "Defensive Coverage"`,
regardless of `player_to_predict`.

For cross-boundary separation at horizon `h`, the evaluable defender set is
restricted to defensive entities that:

- are observed at the prediction origin;
- have `player_to_predict == True`;
- have an unambiguous, matched output group; and
- contain valid, contiguous output frames `1..h`.

Future separation is therefore separation within the **supplied
output-evaluable defender subset**, not separation from every defender who
might have been on the field.

### 3.2 Inclusion and exclusion ledger

Every rule is applied without reference to model error or difficulty. The
cohort builder must retain a boolean flag for every applicable rule and one
ordered primary exclusion reason so counts reconcile without double-counting.

| ID | Rule and rationale | Affected unit and exclusion level | Expected effect from Milestone 1 | Status |
|---|---|---|---|---|
| C01 | Require a source input play and matching supplementary `(game_id, play_id)`. Geometry without play metadata cannot be interpreted or split. | Play; exclude the play from every cohort. | 0 of 14,108 tracked plays. | Scientifically required |
| C02 | Require non-null, parseable `game_id`, `play_id`, `nfl_id`, and integral positive `frame_id`. | Invalid entity-frame; if the target is affected, exclude the target-frame/sequence; if all defenders are affected, exclude the target-frame. | 0 missing `nfl_id` rows and no reported key failures. | Scientifically required |
| C03 | Require uniqueness of `(game_id, play_id, phase, frame_id, nfl_id)`. No arbitrary duplicate resolution is allowed. | Duplicate entity sequence; propagate to target-frame or play only when the target or all defenders become unusable. | 0 input and 0 output duplicates. | Scientifically required |
| C04 | Require exactly one `Targeted Receiver` `nfl_id` per play and one target row per target-frame. | Play for conflicting target identities; target-frame for a missing/duplicate target row. | 14,108 of 14,108 plays satisfy the rule. | Scientifically required by the scoped question |
| C05 | Require one internally constant `play_direction` value in `{"left", "right"}` per play. | Play; exclude from normalized geometry until direction is resolved. | 0 direction-conflict plays. | Scientifically required |
| C06 | Require finite target coordinates. Require finite defender coordinates for each retained pair. Aggregate nearest/kth/count quantities additionally require complete coordinates for every registered observed defender so missing tracking cannot masquerade as space. | Missing target: target-frame. Missing defender: pair-frame; invalidate aggregate target-frame quantities while retaining valid pair-specific quantities. | 0 missing input and 0 missing output coordinate rows. | Scientifically required |
| C07 | Require at least one valid observed defender for target-centric geometry. Do not impose a larger minimum merely for convenience. | Target-frame. Do not discard the entire play when other frames remain valid. | Exclude exactly 35 of 396,914 descriptive target-frames (0.0088%); primary-origin impact remains to be measured. | Scientifically required |
| C08 | Retain and flag coordinates inside the extended boundary tolerance; never clip. Coordinates beyond the extended boundary invalidate only the affected entity-frame and dependent samples. | Entity-frame, then pair/target-frame as required. | Three consecutive Week 3 targeted-receiver output rows contain the four flagged coordinate cells; all are inside tolerance and are retained. | Scientifically required handling; tolerance is a declared convention |
| C09 | Require consecutive frame IDs for every history or future sequence actually consumed. A gap outside the consumed interval does not invalidate unrelated frames. | Entity-history or future sequence; target sample only if its required target/defender sequence fails. | 0 input gaps/regressions/duplicates and 0 noncontiguous output groups. | Scientifically required |
| C10 | Require one non-null positive `num_frames_output` declaration per prediction candidate, a matching output group, and exact frames `1..num_frames_output`. | Output entity sequence. Descriptive input geometry remains eligible. | 0 conflicts, 0 missing groups, and 0 declared-frame mismatches across 46,045 groups. | Scientifically required for cross-boundary tasks |
| C11 | Use `player_to_predict` only to identify supplied prediction/evaluation trajectories. All valid input entities may provide context. | Entity task assignment, not an exclusion from descriptive context. | Targets are always prediction candidates; some defensive-coverage entities are candidates; passers and other route runners are not. | Dataset-imposed |
| C12 | Require 2 consecutive input frames ending at the fixed origin for the common constant-position/constant-velocity baseline cohort. For future separation, this applies to the target and each evaluable defender; remove a defender lacking the history and exclude the target-origin only if none remains. Materialize a separate 5-frame history-eligible flag for possible later engineered histories, but do not require it for M2 baselines. | Entity-origin or target-origin sequence, not whole play. | Unknown until cohort construction; must be reported for both 2- and 5-frame flags by role, week, and split. | Two frames are scientifically required by the declared velocity; five is a postponed methodological convention |
| C13 | Evaluate `h in {5, 10, 15}` only when frames `1..h` exist for every required target/defender output sequence. A missing future is not a negative label. | Horizon-specific entity-origin or target-origin sample. | Declared output length is at least 5 for all audited groups; reductions at 10/15 frames and defender-set reductions are unknown until cohort construction. | Scientifically required; horizon choices are conventions |
| C14 | Join output rows to input role/side by `(game_id, play_id, nfl_id)` and require one unambiguous role/side assignment within the play. | Output entity sequence. | Expected to be zero exclusions; exact consistency must be asserted by the cohort builder. | Scientifically required |
| C15 | Do not exclude rare roles, short-but-eligible futures, large errors, cuts, boundary plays, incompletions, or other difficult examples because they hurt performance. | All units. | No direct reduction; protects the study population from performance-driven filtering. | Scientifically required |

Primary exclusion reasons use C01 through C14 order; C15 is an audit
assertion. The ledger must also retain all secondary reason flags. Counts must
be reported separately for plays, target-frames, pair-frames,
entity-origins, and each horizon.

## 4. Coordinate transformation contract

### 4.1 Raw evidence and units

The working field reference is `L = 120.0` on the long axis and `W = 53.3` on
the cross-field axis. Observed raw ranges were:

| Phase | x range | y range | Missing |
|---|---:|---:|---:|
| Input | 0.41–119.86 | 0.62–52.88 | 0 rows |
| Output | 0.02–120.83 | 0.33–53.72 | 0 rows |

The values are internally consistent with NGS-style yard coordinates, but the
local release lacks a retained authoritative coordinate dictionary. Until
that documentation is pinned, schemas must label spatial units
`source_field_unit`; reports may parenthetically call them yard-equivalent
only with the documentation citation. Frame-difference velocity is expressed
in source field units per frame. Source `s` and `a` are not used to infer time
units until their definitions are authoritative.

Raw `x`, `y`, `dir`, and `o` are immutable audit columns. Normalized fields use
new names and include transform version, raw-boundary flag, and normalized-
boundary flag.

### 4.2 Common attacking direction

The working interpretation is that `x` is the long field axis, `y` is the
cross-field axis, `play_direction == "right"` attacks toward increasing raw
`x`, and `"left"` attacks toward decreasing raw `x`. Milestone 1 found both
values, no within-play conflicts, field-like coordinates, and strong agreement
between `dir` and observed motion. The interpretation remains
documentation-dependent and must be named as such.

Use a 180-degree rotation for leftward plays so every normalized offense
attacks toward increasing `x`:

| Raw direction | Position transform | Angle transform |
|---|---|---|
| `right` | `x_norm = x`; `y_norm = y` | `dir_norm = dir mod 360`; `o_norm = o mod 360` |
| `left` | `x_norm = L - x`; `y_norm = W - y` | `dir_norm = (dir + 180) mod 360`; `o_norm = (o + 180) mod 360` |

For explicit vectors, rightward plays use `(vx_norm, vy_norm) = (vx, vy)` and
leftward plays use `(-vx, -vy)`; acceleration vectors follow the same rule.
Scalar speed and acceleration magnitude do not change. Angle value 360 wraps
to 0 in normalized fields, while the raw value is retained.

Milestone 1's best motion-angle candidate was
`(vx, vy) proportional to (sin(dir), cos(dir))`, with median direction/motion
cosine 0.99955. This is strong internal evidence, not an authoritative angle
definition. Position-difference velocities are the primary M2 velocities.
`dir` and `o` may enter geometry or models only after a documented semantic
check.

The leftward transform is its own inverse. Applying it twice must recover raw
positions and canonical raw angles (`angle mod 360`) within `1e-5` source field
unit/degrees. It must preserve all pairwise distances and vector norms within
`1e-5`.

### 4.3 Boundary policy

Nominal bounds are `[0, L] x [0, W]`. The explicit audit tolerance is
`delta = 1.0` source field unit, giving extended bounds
`[-1, 121] x [-1, 54.3]`. This is a fixed plausibility envelope for tracked
player centers immediately beyond a sideline/endline; it also covers the
observed Week 3 output maxima (x=120.83, y=53.72). It may not expand
automatically in response to later data.

- Values inside nominal bounds are retained without a boundary flag.
- Values outside nominal but inside extended bounds are retained and flagged.
- Values outside extended bounds are not clipped or repaired. Exclude the
  affected entity-frame from geometry and propagate the exclusion according
  to C06/C07.
- Reports must count affected coordinate cells and unique entity-frames by
  phase, week, role, and split.
- Plotting limits may extend to the tolerance; they must not visually snap a
  point to the field boundary.

The four known Week 3 output coordinate cells occur in exactly three
consecutive targeted-receiver rows:
`(game_id=2023092411, play_id=2666, nfl_id=46213, output frame_id=15..17)`,
at `(120.26, 52.12)`, `(120.57, 52.96)`, and `(120.83, 53.72)`. They form a
continuous trajectory, are retained and flagged, and must appear in a boundary
error slice.

## 5. Temporal-index contract

Frame order, not wall-clock time, is the primary temporal index. Milestone 1
observed 4,707,429 consecutive input transitions of exactly `+1`, with zero
gaps, duplicates, or regressions. No timestamp or event field exists.

For a play, define `T` as the maximum raw integral input `frame_id` before
coordinate or cohort exclusions. Never move the origin backward because an
entity is invalid at `T`; exclude that entity-origin instead. Then:

- input `relative_frame = frame_id - T`, so the primary origin is 0;
- output `relative_frame = frame_id`, so its first supplied future is 1; and
- `phase` remains part of every key even after relative indexing.

This defines the **supplied sequence boundary** between relative frames 0 and
1. It must not be described as an observed timestamp. The primary prediction
origin is the last supplied input frame, one origin per play. Rolling
within-input prediction origins are postponed; descriptive input geometry may
still be calculated at every eligible target-frame.

The common M2 baseline history is 2 consecutive input frames ending at
relative frame 0; a separate 5-frame eligibility flag is reserved for later
engineered histories. Primary horizons are 5, 10, and 15 output frames. All
methods and artifacts must name them in frames. They may additionally be
labeled approximately
0.5, 1.0, and 1.5 seconds only after an authoritative 10-Hz source is cited;
the code must accept cadence as metadata rather than hard-code it. Calculations
that use positions, frame differences, ranks, or frame counts remain valid
without conversion to seconds.

`num_frames_output` is an offline label-availability field. It may determine
whether a candidate-horizon can be evaluated, but may not be a predictive
feature. No padding, extrapolated label, or negative label is created when a
horizon is unavailable.

## 6. Entity-role semantics

| Semantic role | Source rule | Permitted use |
|---|---|---|
| Targeted receiver | The unique play-level `nfl_id` with `player_role == "Targeted Receiver"` | Condition defining the target-centric task, target trajectory, and target-frame geometry. This does not model target selection. |
| Passer | `player_role == "Passer"` and offense side | Contemporaneous input context only. No passer trajectory label is supplied under the core target task. |
| Other route runner | `player_role == "Other Route Runner"` and offense side | Contemporaneous input context; never evidence that all eligible receiving options are present. |
| Defensive player | `player_side == "Defense"` and `player_role == "Defensive Coverage"` | All valid same-frame defenders enter descriptive geometry. Output-available defensive candidates enter cross-boundary separation evaluation. |
| Prediction candidate | `player_to_predict == True` with a valid expected/matched output group | May become a trajectory target or output-evaluable defender. The flag is a cohort selector, not a numeric feature. |
| Contextual entity | Any valid input entity observed no later than the origin, including non-prediction candidates | May be a model input subject to leakage and missingness rules; has no implied future label. |

Defenders are associated with the target by same-play, same-frame observation
and deterministic geometry. Nearest defender means the smallest Euclidean
distance within the declared defender set. It is not an official coverage
assignment. All observed defenders may be used without asserting man/zone
responsibility, matchup, leverage, or complete defensive control.

## 7. Leakage policy

Every feature record must store `game_id`, `play_id`, split, origin, maximum
source frame used, label phase, label interval, and feature provenance. The
feature builder must assert that no feature source exceeds the origin.

| Field or information | Retrospective descriptive use | Prediction-time feature | Label/evaluation use | Binding rule |
|---|---|---|---|---|
| Input positions and origin-available roles at or before origin | Allowed | Positions are allowed; defensive/passer/other-route roles may provide context. Target designation is conditioning/selection, not a variable feature. | Allowed | Use only declared trailing history. Raw `nfl_id` is for joins/reporting, not a default feature. |
| Input frames after an earlier retrospective cutoff | Allowed when explicitly tagged retrospective | Forbidden | May define an explicitly separate within-input label | Never enter features for that cutoff. Primary cross-boundary origin has no later input feature frames. |
| Output coordinates | Allowed in post-boundary descriptions | Forbidden | Allowed as trajectory and separation labels | Evaluation only for the primary prediction task. |
| Future separation or separation change | Allowed after the fact | Forbidden | Allowed | Derived label; never recompute it inside a feature transform. |
| Future nearest-defender identity | Allowed after the fact | Forbidden | Allowed as an evaluation annotation | Current nearest identity may be derived at the origin; future identity may switch and cannot be leaked. |
| Competition target role/ID | Allowed | Allowed only to select the known conditioned target; ID value itself forbidden as a default feature | Allowed | The task asks about a known target and makes no QB target-selection claim. |
| `player_to_predict` | Allowed for source accounting | Forbidden as a numeric/categorical model feature | Allowed only for cohort and label-availability selection | It reflects competition packaging and future-label availability. |
| `ball_land_x`, `ball_land_y` | Audit or frozen post-hoc slicing only | Forbidden | Not a primary M2 label | Prohibited from cohort geometry and baseline features. |
| `num_frames_output` | Allowed for audit | Forbidden | Allowed only to establish evaluable horizons | Do not expose its value to a model. |
| Pass result, completion, final outcome, penalties, yards gained | Frozen post-hoc description only | Forbidden | Not a primary trajectory/separation label | Keep out of cohort selection except a predeclared source-integrity rule. |
| Route/role/selection information known only after origin | Frozen post-hoc description only | Forbidden | Evaluation annotation only | Availability must be proven at the origin before any feature use. |
| Full-play or future-window aggregates | Allowed in retrospective reports | Forbidden | Allowed as explicitly named labels | Trailing aggregates must be cutoff-aware; centered/full-play smoothing is forbidden. |
| Source `s`, `a`, `dir`, `o` at/before origin | Allowed with caveat | Allowed only after semantics are verified | Allowed | Prefer trailing position differences until authoritative definitions are pinned. |
| Imputation, scaling, thresholds, feature selection, calibration statistics | Descriptive raw summaries may use the declared population | Training-fitted values only | Validation/test are transform-only | Fit within each training fold; never use validation/test distributions to set values. |

Landing coordinates, final outcomes, future roles, and future-derived
aggregates are prohibited entirely from the primary M2 feature table. They
remain in immutable raw/audit data only.

## 8. Geometric definitions

Let `p_r(f)` be the normalized target position and `p_d(f)` a normalized
eligible-defender position. Define the target-to-defender vector
`q_d(f) = p_d(f) - p_r(f)` and pair distance
`d_d(f) = ||q_d(f)||_2`. Position and distance units are source field units.
Define trailing velocity from positions as
`v_i(f) = p_i(f) - p_i(f-1)`, in source field units per frame.

| Quantity | Formula and sign convention | Required fields, missing behavior, and edge cases | Evidence level |
|---|---|---|---|
| Nearest-defender distance | `d_(1)(f) = min_d d_d(f)` | Target plus at least one eligible defender at the same frame. Missing when no defender remains. Ties use the lowest canonical `nfl_id`. Finite and nonnegative. | Deterministically derived |
| kth-nearest distance | Sort by `(distance, canonical nfl_id)`; return `d_(k)(f)` | Missing when fewer than `k` eligible defenders exist; do not exclude the frame from lower-order metrics. Report available defender count with every value. | Deterministically derived |
| Local defender count | `N_R(f) = sum_d 1[d_d(f) <= R]` | Primary `R=5`; sensitivity radii 3 and 10 source field units. Boundary is inclusive. Count is 0 only on a retained frame with defenders all outside `R`; no-defender frames are excluded, not assigned 0. | Deterministically derived |
| Relative position | `q_d = p_d - p_r` | Components point from target to defender. Missing if either position is invalid. | Deterministically derived |
| Relative velocity | `u_d = v_d - v_r` | Requires consecutive positions for both entities. Pair is missing at the first usable frame or across a gap; no interpolation. | Deterministically derived |
| Radial closing velocity | Distance derivative `dot_d = (q_d · u_d) / d_d`; closing `c_d = -dot_d` | Positive `c_d` means closing, negative means separating, zero means no radial change. Undefined at `d_d=0`; flag collision and return missing rather than divide by epsilon. Units per frame. | Deterministically derived |
| Separation change | `Delta_h = d_(1)(f+h) - d_(1)(f)` | Positive means more separation; negative means closing. Current and future nearest defenders may differ, and both identities must be retained. Missing if either target-frame is unevaluable. | Deterministically derived label |
| Separation persistence | `P_(tau,q)(f)=1` iff `d_(1)(u) >= tau` for every `u=f..f+q-1` | `tau` and persistence length `q` are configuration values frozen using training data and predeclared sensitivity analysis. Any missing frame makes the label unavailable, not false. | Deterministically derived label |
| Frame to minimum separation | `u* = min argmin_(0<=u<=h) d_(1)(f+u)`; also retain the minimum distance | Earliest frame wins a tie. Units are frames; missing if the complete interval is unavailable. Nearest identity may change. | Deterministically derived |
| Constant-position baseline | `p_hat_i(f+h) = p_i(f)` | Requires only the origin position, but is evaluated on the same 2-frame-history cohort as constant velocity for a fair comparison. Never clipped to the field. | Deterministic model-dependent baseline |
| Constant-velocity baseline | `p_hat_i(f+h) = p_i(f) + h * v_i(f)`, with one-frame trailing `v_i` | Requires consecutive frames `f-1,f`; the common cohort requires exactly this minimum. No future/source `s` is used. Predictions outside tolerance are retained and flagged, not clipped. | Deterministic model-dependent baseline |

Predicted future separation is computed by applying the same distance
definitions to predicted target and defender positions over the same declared
output-evaluable defender set. Changing that set between baselines is
forbidden.

`tau`, persistence `q`, and whether the 5-unit local radius remains primary
must be frozen from training-only descriptive evidence before baseline
comparison. Their football interpretation is geometric only.

## 9. Evaluation split strategy

The only locally labeled tracking season is 2023, so cross-season
generalization is not measurable. The primary split is chronological and
week-blocked:

| Split | Weeks | Audited games/input rows/output rows | Use |
|---|---|---:|---|
| Development train | 1–12 | 180 / 3,236,116 / 368,514 | Cohort diagnostics, train-only parameter choices, and fitting. |
| Validation | 13–15 | 44 / 795,547 / 95,156 | Horizon/threshold checks and baseline selection. May not influence train-fitted transforms. |
| Frozen test | 16–18 | 48 / 848,916 / 99,266 | One final evaluation after code, cohorts, horizons, and metrics are frozen. |

Within weeks 1–12, use expanding development folds:
train 1–6/validate 7–8, train 1–8/validate 9–10, and train
1–10/validate 11–12. A game and all of its plays, frames, pairs, entities, and
horizons belong to exactly one split. Random row-, frame-, play-, or
nonchronological group splitting is prohibited.

Players and teams may recur across splits; this is the natural later-week
prediction setting. Player/team identity is excluded from default predictive
features, and seen/unseen or frequency slices are robustness diagnostics, not
alternate primary splits. Any imputation, normalization statistics,
thresholds, feature selection, or later calibration are fit within the
training portion. Uncertainty intervals resample games, not entity-frames.

After validation choices are frozen, a baseline may be refit on weeks 1–15
before one evaluation on weeks 16–18. No test result may trigger a parameter,
cohort, threshold, or code-path change without declaring the test compromised
and reserving a new future holdout.

## 10. Acceptance tests

Future cohort, geometry, baseline, and split code must satisfy all applicable
invariants:

### Cohort and keys

1. Raw-universe counts reconcile to 14,108 plays, 4,880,579 input rows,
   562,936 output rows, 396,914 target-frames, and 46,045 output groups before
   exclusions.
2. Every derived entity-frame key includes `phase` and is unique.
3. Every retained target-frame has exactly one target and at least one eligible
   defender; the 35 known zero-defender input frames are counted as exclusions.
4. Every exclusion is counted by unit, rule, week, role, and split; primary
   reasons sum to the source denominator while secondary flags remain
   available.
5. Every output role/side joins unambiguously to its play-level input entity.
6. A candidate at horizon `h` has exactly frames `1..h`; absent futures create
   no labels.

### Coordinates and geometry

7. Rightward normalization is identity; applying the leftward transform twice
   reconstructs coordinates/angles within `1e-5`.
8. Pairwise distances, vector norms, entity ordering, and frame ordering are
   invariant under normalization within `1e-5`.
9. Synthetic left/right mirror plays normalize to the same geometry.
10. No raw or normalized coordinate is silently clipped. The Week 3 output
    anomalies are retained, flagged, and counted.
11. Every pair distance is finite and nonnegative; pair count equals the
    eligible defender count for each target-frame.
12. Nearest and kth-nearest ties are deterministic by canonical `nfl_id`.
13. Radial closing velocity is positive for a synthetic closing defender,
    negative for a separating defender, and missing at zero distance.

### Time, leakage, and splits

14. Primary origin relative frame is 0; consumed input frames are `<=0` and
    label frames are output `1..h`.
15. Every feature records source phase and maximum source frame. Every
    `phase="output"` source row is rejected as a feature regardless of its
    numeric frame ID; input rows later than the origin are also rejected.
16. Constant-position and constant-velocity predictions equal the origin at
    `h=0`; constant velocity matches an exact synthetic linear trajectory.
17. No play or game appears in more than one split, and week order is strictly
    chronological.
18. Train-fitted transforms and parameters have no validation/test rows in
    their fit provenance.
19. `player_to_predict`, output length, landing coordinates, future
    coordinates, future nearest identity, and outcome fields are absent from
    the predictive feature matrix.
20. Metrics reconcile from entity/horizon rows to play and game denominators;
    bootstrap resampling units are games.

## 11. Unresolved questions

These questions must be answered or retained as limitations; none authorizes a
silent assumption:

1. Can an authoritative release data dictionary be retained that confirms
   coordinate units, field origin, `s/a/dir/o` semantics, and 10-Hz cadence?
2. Does the last input frame represent the exact release instant or only the
   supplied pre/post sequence boundary?
3. Does the continuous three-frame Week 3 target trajectory represent
   officially valid out-of-bounds player-center motion? It remains retained
   under the declared tolerance regardless.
4. How many plays and target-origins remain under the 2-frame baseline rule
   and optional 5-frame history flag, and how many target-defender sets remain
   evaluable at 5, 10, and 15 frames?
5. Which thresholds `tau` and persistence lengths `q` are stable on training
   data without collapsing prevalence? These are not chosen in this task.
6. Is `num_frames_output` available only as training-label packaging? It is
   treated as unavailable to predictions regardless.
7. Are source role labels constant within every play/entity, including output
   role joins? Milestone 1 strongly supports this but Task 1 must assert it.

## 12. Explicitly postponed

- Implementation of the cohort builder or exclusion ledger.
- Full-dataset coordinate normalization or separation calculation.
- Selection/freeze of separation threshold `tau` and persistence `q`.
- Full-release descriptive results and player summaries.
- Constant-position and constant-velocity execution.
- Any learned model, neural network, graph model, coverage-assignment model,
  quarterback-choice model, or completion model.
- Ball-flight, obstruction, full-field control, and passing-window analysis.
- Polished UI, dashboard, API, Docker, CI, deployment, or additional
  dependencies.
