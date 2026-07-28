# Data Audit — BDB 2026 Analytics Validation

## Source and coverage

- Extracted release: `data/raw/bdb_2026/114239_nfl_competition_files_published_analytics_final`.
- Files validated: 18 weekly input tables, 18 matching output tables, and `supplementary_data.csv`.
- Rows validated: 4,880,579 input, 562,936 output, and 18,009 supplementary.
- Tracking cohort: 272 games and 14,108 `(game_id, play_id)` sequences.
- The zip archive is a packaging artifact and was excluded from table profiling.

## Zero observed structural failures

- All 18 input headers match the observed 23-column schema and all 18 output
  headers match the observed 6-column schema.
- All 14,108 tracked plays join to supplementary metadata; missing input-row
  joins: 0.
- Duplicate `(game_id, play_id, nfl_id, frame_id)` keys: 0 in both input and
  output.
- Input frame regressions: 0; duplicate frame steps: 0; frame gaps: 0.
- Expected-output-frame declaration conflicts: 0. All 46,045 expected output
  groups are present, with 0 frame-count mismatches and 0 noncontiguous
  sequences.
- Missing coordinate rows: 0 in input and 0 in output.
- Input coordinates span x=0.41–119.86 and y=0.62–52.88, with 0 values outside
  the field-like reference bounds.
- Every target-labelled play has exactly one `Targeted Receiver`: 14,108 of
  14,108.

## Observed anomalies

- Of 396,914 target-receiver frames, 35 (0.0088%) have no recorded
  defensive-side entity. These frames require an explicit exclusion rule.
- The weekly manifest identifies the only coordinate-range anomaly in
  `output_2023_w03.csv`: 3 x values and 1 y value exceed the field-like
  reference bounds. The complete output range is x=0.02–120.83 and
  y=0.33–53.72. These post-release positions are retained as observed data,
  not treated as an input-structure failure.
- No week has a schema, key-uniqueness, missing-coordinate, or temporal-order
  failure.

## Checks that could not be performed

- Exact wall-clock sampling and event timing cannot be verified because there
  is no timestamp or event field. Consecutive frame IDs and the
  distance-to-speed ratio are consistent with tracking sequences, but do not
  establish an official sampling frequency.
- Direct ball-path checks cannot be performed because there is no per-frame
  football entity.
- Official units and angular conventions for `x`, `y`, `s`, `a`, `dir`, and
  `o` remain documentation-dependent despite strong internal consistency.
- Player club/team membership cannot be validated at the tracking-row level
  because no such field is present.

## Actual dataset limitations

- The input contains selected route runners, the passer, and
  defensive-coverage entities rather than all 22 players.
- Post-release trajectories cover only `player_to_predict=True` entities, not
  every route runner.
- The local tracking release contains one season, 2023 weeks 1–18.
- `player_to_predict` may identify the target-centric analytic cohort but
  cannot be treated as a predictor of quarterback target selection.
  `player_to_predict` and `ball_land_x/y` must not enter pre-release predictive
  features.

## Scientific interpretation

The release supports target-centric receiver/defender spatial dynamics,
separation and defender closing behavior, and pre-release to post-release
trajectory analysis for the supplied entities.

It does not support defensible claims about all receiving options, complete
defensive space control, quarterback target selection, full-field passing
windows, or direct ball-path obstruction. “Passing window” is reserved for a
later source or extension containing sufficient ball-path and full-field
context.

The single recommended next task is to write the restricted analytic-cohort
and coordinate-contract specification. No Milestone 2 work has been
implemented.
