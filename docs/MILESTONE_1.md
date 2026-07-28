# Milestone 1 — Data Acquisition, Schema Audit, and Spatial Sanity Check

## Purpose and boundary

Milestone 1 answers one question only: **can an authorized source be reliably
reconstructed as a temporally ordered sequence of player locations for the
pre-release portion of passing plays?**

It does not define a passing window, calculate receiver separation, train a
model, tune a model, infer coverage, build an API, add Docker, deploy anything,
or build an application UI. A static field plot or short animation is used
solely to validate schema, coordinates, entity identity, and timing.

The candidate is the [NFL Big Data Bowl 2026 Prediction competition on
Kaggle](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview/abstract).
Its public description says its supplied tracking stops at quarterback release;
NFL Football Operations describes the competition as using 2023–24 data with
predictions evaluated against 2025 Weeks 14–18. [NFL Football Operations — Big
Data Bowl](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl)
Those statements are not a substitute for the downloaded data dictionary.

## Inputs to acquire

### Exact source and acquisition method

Acquire the authorized competition download for
`nfl-big-data-bowl-2026-prediction` after the account holder has accepted the
applicable Kaggle/NFL terms. Use the Kaggle competition download mechanism
(web download or authenticated Kaggle API) and retain the archive unchanged in
`data/raw/bdb_2026/`. Do not scrape, redistribute, or commit source data.

Record in the audit manifest: competition URL, download date, source version if
shown, every original filename, file size, SHA-256, and any access constraint.
The acquisition is incomplete without the competition overview/data dictionary
and any schema/readme supplied in the download.

The exact **filenames** are not locally available today and must not be
invented. The exact logical file set required is:

1. **Tracking table(s)** for the training/available historical plays, including
   every supplied entity and every pre-release frame.
2. **Play metadata table** to identify a game/play, offensive and defensive
   teams, play type, and any pass/outcome/target fields supplied.
3. **Game metadata table** to place each game in season, week, and chronology.
4. **Player reference table** for player identity, position, and display data.
5. **Event/timing information**, whether a separate table or fields within
   tracking, sufficient to locate snap and release or to establish that release
   is the final supplied frame.
6. **Split/label files and sample submission**, if supplied, to determine
   exactly which seasons/games/plays have observed outcomes versus inference
   only.
7. **Data dictionary/readme/rules** defining every field and dataset scope.

If an expected logical table is embedded in another file, document the actual
location. If it is absent, record absence rather than creating a surrogate.

### Expected relationships

The audit will verify—not assume—the following relational keys:

```text
games        1 --- * plays              via game_id
plays        1 --- * tracking frames    via (game_id, play_id)
players      1 --- * tracking entities  via player/nfl_id, when populated
play/events  1 --- * frame events       via (game_id, play_id, frame_id or time)
```

Tracking's retained entity key must distinguish players from ball records. The
preferred key is `(game_id, play_id, frame_id, nfl_id)` for players plus a
documented ball sentinel/role. If player ID is unavailable for some entities,
use the source's documented entity key and flag the limitation.

## Schema expectations and required verification

“Expected” below means common NGS-style semantics or a public competition
description; it is not a claim about the locally absent files. “Required”
means the audit cannot advance without a documented answer.

| Logical table | Expected fields/semantics | Must verify in M1 |
| --- | --- | --- |
| Tracking | Game/play/frame identifier; entity/player ID; team/side or club; `x,y`; possibly speed, acceleration, direction, orientation, event, and display name. | Exact columns/dtypes, units, location of ball rows, valid key, whether fields are measured or precomputed, and whether all relevant players or only a subset are tracked. |
| Plays | Game/play ID; offense/defense; play type/description; down/distance; possibly pass result, targeted receiver, landing location, and release/arrival metadata. | Authoritative pass-play flag, team-side definitions, receiver/target fields, result fields, and whether fields are available before release or are post-play labels. |
| Games | Game ID, season, week, date, home/visitor teams. | Stable chronological ordering and alignment to every play. |
| Players | Player/NFL ID, position, name, possibly height/weight. | ID join rate, position vocabulary, time relevance of the reference table, and treatment of missing IDs/positions. |
| Events/timing | Frame index and/or timestamp; snap/release events or an endpoint convention. | Event vocabulary, event-to-frame alignment, cadence, whether timestamps are monotonic, and an authoritative pre-release interval. |
| Split/labels | Train/test indicator, labels/outcomes, target/landing information, sample submission. | Which rows contain labels, whether held-out files can be evaluated locally, and exact train/validation/test chronology. |

### Tracking sampling and timing

NGS tracking is often described at 10 Hz, but M1 must compute the observed
cadence from the delivered data. For each retained play, report frame counts,
inter-frame timestamp differences when timestamps exist, and the distribution
of frames per second. If only frame IDs exist, determine whether their step is
one per sample and validate the effective rate from documentation; do not
assume 10 Hz in code or reporting.

Identify, in order of preference:

1. documented snap and release events aligned to frame IDs;
2. a documented final-pre-release-frame convention; or
3. no reliable release boundary (a limitation).

No free-text play-description parser may be the primary timing source. It can
only cross-check an event field. The retained M1 sequence is from the first
valid observed frame through the documented release boundary; no future frame
is needed or used.

### Coordinates and direction

Do not normalize coordinates in Milestone 1. First establish the raw
convention:

- expected location columns and units (usually yards, but verify);
- coordinate origin, axes, field bounds, and whether end zones are included;
- offense direction field or the information necessary to derive it;
- definitions and angular units/conventions for direction/orientation, if
  present; and
- whether velocities are source fields or future-looking/smoothed derivatives.

As a plausibility reference only, full-field NGS-style coordinates often fall
near `x in [0, 120]` and `y in [0, 53.3]` yards. M1 must derive acceptable
bounds from the delivered dictionary, allow a small documented tolerance for
ball/end-zone observations, and flag—not clip—out-of-range values. The output
must explicitly say whether a single raw axis corresponds to downfield motion
in both play directions.

### Identifying passing plays and relevant entities

Use the source's structured pass-play/type/result field as the authoritative
selection mechanism. Cross-check a stratified sample against play description
and event sequence. Report the count excluded as runs, sacks, scrambles,
penalties/no-plays, missing event boundaries, or ambiguous play types; retain
no ambiguous row silently.

For each retained play, determine and document:

- offensive and defensive team identity;
- all tracked player entities, the ball if present, and their side/club;
- player position when joined to the reference table;
- first valid frame and release boundary; and
- whether the table contains all 22 players plus a ball, a subset, or another
  competition-specific entity set.

The public competition overview says a targeted offensive player and pass
landing location are provided. M1 must verify the actual field names, null
rate, and train/test availability. Target information is an audit field only;
it is not used to define a receiver set, select frames, or score a model in
this milestone.

## Required validation checks

Produce counts, rates, and a list of affected `(game_id, play_id)` values for
each check. Never silently drop or repair data in the audit.

| Check | Required procedure and acceptance rule |
| --- | --- |
| File integrity | Hash and inventory all source files; parse every file with explicit dtype/error reporting. Every parsed table has a documented row count. |
| Referential integrity | Measure join coverage from tracking to plays/games/players. No retained tracking row may lack a game/play join; player-ID join failures may remain only if documented and do not prevent entity-side identification. |
| Missing data | Tabulate missingness overall and by table/split for every key, time, entity/side, coordinate, direction, event, player-position, and target field. Coordinates or temporal keys missing in a retained frame are disqualifying; M1 performs no imputation. |
| Duplicates | Count exact duplicates and duplicate entity records under the documented tracking primary key. Retained rows require exactly one record per entity per frame; duplicates are resolved only by a source-documented rule or make the play invalid. |
| Temporal order | Within play and entity, frame IDs/timestamps must be strictly increasing with no unexplained reversal. Check cross-entity alignment, positive/interpretable frame intervals, snap <= release, and no retained frame after release. |
| Coordinate range | Summarize min/max/quantiles of raw coordinates by entity type; flag points outside documented field/tolerance bounds, discontinuities, and impossible all-entity jumps. No clipping. |
| Entity/player counts | Report player and ball counts per frame/play, separately for offense/defense. Verify expected 22-player-plus-ball coverage **or** precisely document the competition's subset. A receiver/defender analysis cannot claim all-receiver opportunity if the necessary defenders or receivers are absent. |
| Team/side consistency | Check that tracked clubs match play offense/defense fields and that a player does not appear on conflicting sides in a frame without a source explanation. |
| Event semantics | Tabulate event values and their frames; manually verify snap/release semantics on selected plays. Absence of a release event is a named limitation, not inferred from the final row without documentation. |
| Chronology/splits | Confirm every game has a season/week/date order, no game is in multiple official splits, and labels are never mistaken for available inference-time fields. |

## Manual spatial sanity check

Select twelve plays after automated checks using a recorded random seed:

- eight valid passing plays stratified by offensive direction, early/late game
  chronology, and available pass-depth/field-location bins; and
- four edge cases: a short pass, a deeper pass, a sideline-oriented play, and a
  play near a boundary or with an unusual/missing event pattern. If a requested
  category is unavailable, record the substitution rather than hand-picking a
  highlight.

For each selected play, make a minimal static plot with the raw field axes,
offense/defense/ball color legend, player labels where available, and three
ordered frames (start, middle, last retained). Make one short raw-coordinate
animation per play at the verified cadence or frame sequence. It must show no
derived separation, no prediction, and no UI controls beyond what is necessary
to inspect the sequence.

Manually answer and record for every selected play:

1. Do player/ball locations lie on a plausible field and move continuously?
2. Do colors/sides match the play metadata?
3. Does the sequence advance in the documented temporal order?
4. Does the retained endpoint agree with the source's release/endpoint rule?
5. Is the raw direction convention understandable enough to normalize later?

Any unexplained failure is a failed validation item, not a visualization issue.

## Deliverables

Milestone 1 produces exactly these artifacts; it produces no model artifact:

1. `docs/data_audit.md` — source inventory, schema/data dictionary summary,
   confirmed versus unknown fields, joins, validation tables, exclusions, and
   data-use constraints.
2. `artifacts/data_manifest.json` — source filenames, sizes, hashes, download
   metadata, and row counts; no source rows.
3. `artifacts/milestone_1_validation.json` — machine-readable results for
   missingness, duplicates, joins, cadence, ranges, entity counts, and timing
   checks.
4. `notebooks/01_data_audit.ipynb` (or an equivalently reproducible script) —
   read-only audit and the twelve selected-play checks.
5. `artifacts/milestone_1_sanity/` — twelve static panels and twelve short
   raw-coordinate animations, plus a CSV/JSON listing play IDs and selection
   strata.
6. A one-page decision note appended to `docs/data_audit.md`: GO, LIMITED GO,
   or NO GO, with evidence against each exit criterion below.

Raw files stay in `data/raw/`; derived audit summaries contain no restricted
tracking rows unless the source terms explicitly allow them.

## Exit decision

### GO

Declare **GO** only if all of the following hold for a substantial, documented
set of structured passing plays:

- tracking rows join reliably to play and game metadata;
- each retained play is a temporally ordered sequence with usable coordinates;
- offense/defense entity identity is reliable and enough tracked entities are
  present to support the intended receiver/defender geometry (normally all
  relevant defenders, not merely a target and one opponent);
- cadence and raw coordinate conventions are verified or have an authoritative
  documented interpretation;
- a defensible pre-release endpoint is identified; and
- the twelve visual checks have no unexplained coordinate, side, timing, or
  entity-identity error.

“Substantial” must be reported as both a play count and a fraction of the
structured pass-play source. It is not satisfied by a hand-selected few plays.

### LIMITED GO

Declare **LIMITED GO** when locations, time order, and enough entities support
meaningful spatial analysis, but a verified limitation forces a narrower
question. Examples: only the targeted receiver and a subset of defenders are
tracked; no reliable release event exists but an ordered within-play interval
does; player position is incomplete; or target/landing labels are unavailable
in the usable split.

The decision note must state the revised question and forbidden claims. For
example, subset tracking permits a *recorded receiver–defender interaction*
study, not an all-receiver opportunity or passing-window study.

### NO GO

Declare **NO GO** for this source if any critical condition remains unresolved:

- tracking cannot be reliably joined to game/play metadata;
- coordinate locations or timing cannot be interpreted and visually verified;
- offense/defense identities cannot be determined for relevant entities;
- duplicate, missing-coordinate, or temporal-order errors affect the retained
  sample without a source-supported resolution; or
- the source lacks enough jointly tracked receiver and defender positions to
  make even the narrowed geometric question scientifically defensible.

A NO GO stops downstream feature/model work. The next action is a documented
source pivot, not imputation, heuristics, or a more complex model.

## First task after approval

Implement one read-only **source inventory and schema profiler** that scans the
authorized download and writes the manifest plus column, dtype, key, and row
count report. It is intentionally the first task because every later M1 check
depends on the actual filenames and schema it establishes.
