# Milestone 1 Result

## Decision

**LIMITED GO**

The release is structurally reliable for target-centric receiver/defender
analysis, but its partial entity coverage and lack of per-frame football
tracking require a narrower research question. In addition, 35 of 396,914
target frames (0.0088%) contain no recorded defender and require exclusion.

## Full-release execution

The finalized validator ran across all 18 weekly input/output pairs in one
complete execution.

| Stage | Runtime |
|---|---:|
| Supplementary metadata | 0.277 s |
| Input validation | 260.021 s |
| Output validation | 9.303 s |
| Deterministic play selection | 0.117 s |
| Raw-coordinate SVGs | 21.280 s |
| **Total measured stages** | **290.998 s** |

Peak resident memory was 694,476 KiB (about 678 MiB).

## Aggregate coverage

| Quantity | Count |
|---|---:|
| Weekly input/output pairs | 18 |
| Input rows | 4,880,579 |
| Output rows | 562,936 |
| Supplementary rows | 18,009 |
| Tracking games | 272 |
| Tracked game/play pairs | 14,108 |
| Target-labelled plays | 14,108 |
| Target-receiver frames | 396,914 |
| Expected/observed output player groups | 46,045 / 46,045 |

The supplementary table contains 18,009 unique game/play pairs. All tracked
plays join to it; the remaining 3,901 metadata-only pairs are outside the
tracked input cohort.

Input role counts sum exactly to the input-row count:

| Role | Rows |
|---|---:|
| Defensive Coverage | 2,662,657 |
| Other Route Runner | 1,424,243 |
| Passer | 396,765 |
| Targeted Receiver | 396,914 |

## Structural evidence

Zero failures were observed for:

- weekly schema consistency: 18 of 18 input and 18 of 18 output files match;
- input and output entity-frame key uniqueness: 0 duplicate rows;
- input-to-supplementary joins: 0 unmatched input rows;
- output-to-input declarations: 0 unmatched rows and 0 missing groups;
- input frame order: 0 regressions, 0 duplicates, and 0 gaps across
  4,707,429 observed `+1` transitions;
- expected-output declarations: 0 conflicts;
- output frame structure: 0 expected-count mismatches and 0 noncontiguous
  groups; and
- coordinate missingness: 0 input rows and 0 output rows.

The candidate entity-frame keys are therefore unique for all 4,880,579 input
rows and all 562,936 output rows.

## Coordinates and weekly anomalies

- Raw input ranges are x=0.41–119.86 and y=0.62–52.88, with 0 values outside
  the field-like reference bounds.
- Raw output ranges are x=0.02–120.83 and y=0.33–53.72.
- Week 03 is the only weekly coordinate anomaly: its output has 3 x values and
  1 y value beyond the reference bounds. No weekly schema, key, missingness, or
  temporal failure was observed.
- No coordinate transform was applied. The official coordinate and angular
  conventions remain documentation-dependent.

## Entity coverage

Every one of the 14,108 target-labelled plays has exactly one targeted
receiver. Defensive entities per target frame have min=0, p01=4, p05=5,
median=7, p95=7, p99=8, and max=11. The 35 zero-defender frames are 0.0088% of
396,914 target frames.

Total tracked entities per frame:

| Entities | Frames | Entities | Frames |
|---:|---:|---:|---:|
| 1 | 35 | 10 | 21,453 |
| 5 | 23 | 11 | 49,005 |
| 6 | 51 | 12 | 91,614 |
| 7 | 640 | 13 | 210,838 |
| 8 | 1,639 | 14 | 14,069 |
| 9 | 6,985 | 15 | 322 |
| 16 | 57 | 17 | 183 |

Counts of 2–4 do not occur. These 396,914 frame groups demonstrate that the
release is a selected tracking subset, not full-22 tracking.

## Checks that could not be performed

- Exact wall-clock timing and official sampling frequency: no timestamp/event
  field.
- Direct football-path reconstruction or obstruction: no per-frame football
  entity.
- Tracking-row club/team validation: no individual club field.
- Cross-season generalization: only 2023 weeks 1–18 are present locally.

These are source limitations, not observed schema or key failures.

## Scientific interpretation

The dataset supports:

- target-centric receiver/defender spatial dynamics;
- separation and defender closing behavior; and
- pre-release to post-release trajectory analysis for prediction-designated
  entities.

It does not support defensible claims about:

- all receiving options;
- complete defensive space control;
- quarterback target selection;
- full-field passing windows; or
- direct ball-path obstruction.

“Passing window” is reserved for a later dataset or extension containing
sufficient ball-path and full-field context.

The narrowed research question is:

> For a competition-designated target and the observed coverage defenders, how
> do separation and closing behavior evolve from pre-release geometry into the
> supplied post-release trajectory?

Exactly 12 deterministic raw-coordinate sanity SVGs are stored in
`artifacts/milestone_1_sanity/`; their selection metadata is in
`artifacts/milestone_1_validation.json`.

## Recommended next task

Write the restricted analytic-cohort and coordinate-contract specification:
competition-designated targets and their observed defensive context,
pre-release input frames plus explicitly scoped post-release trajectories, a
rule excluding the 35 zero-defender frames, and a prohibition on using
`player_to_predict` or ball-landing labels as predictive features. This task
has not been implemented.
