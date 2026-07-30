# Milestone 2 Cohort Result

## 1. Task status

Milestone 2 Task 1, the analytic cohort builder and artifact pipeline, is
complete. Final status: **PASS within the Milestone 1 LIMITED-GO scope**.

## 2. Source cohort

- Processed weeks: `2023_w01` through `2023_w18`
- Input rows: 4,880,579
- Output rows: 562,936
- Unique tracking games: 272
- Source plays: 14,108

## 3. Cohort counts

| Cohort table | Rows | Eligible | Excluded |
|---|---:|---:|---:|
| `source_plays` | 14,108 | 14,108 | 0 |
| `descriptive_target_frames` | 396,914 | 396,879 | 35 |
| `primary_origins` | 14,108 | 14,108 | 0 |
| `trajectory_eligibility` | 138,135 | 85,542 | 52,593 |
| `future_separation_eligibility` | 42,324 | 23,100 | 19,224 |
| `pair_exclusions` | 0 | 0 | 0 |

The exclusion ledger contains 71,852 rows.

## 4. Split counts

Each cell reports rows / eligible / excluded.

| Cohort table | Development train | Validation | Frozen test |
|---|---:|---:|---:|
| `source_plays` | 9,400 / 9,400 / 0 | 2,253 / 2,253 / 0 | 2,455 / 2,455 / 0 |
| `descriptive_target_frames` | 262,822 / 262,787 / 35 | 64,737 / 64,737 / 0 | 69,355 / 69,355 / 0 |
| `primary_origins` | 9,400 / 9,400 / 0 | 2,253 / 2,253 / 0 | 2,455 / 2,455 / 0 |
| `trajectory_eligibility` | 91,848 / 56,071 / 35,777 | 22,392 / 14,353 / 8,039 | 23,895 / 15,118 / 8,777 |
| `future_separation_eligibility` | 28,200 / 15,207 / 12,993 | 6,759 / 3,832 / 2,927 | 7,365 / 4,061 / 3,304 |
| `pair_exclusions` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Ledger rows are split 48,805 development train, 10,966 validation, and
12,081 frozen test.

## 5. Horizon eligibility

Future-separation eligibility follows directly from the generated
evaluable-defender distributions:

| Horizon | Candidate target-horizons | Eligible | Excluded |
|---|---:|---:|---:|
| 5 frames | 14,108 | 12,966 | 1,142 |
| 10 frames | 14,108 | 7,404 | 6,704 |
| 15 frames | 14,108 | 2,730 | 11,378 |

The generated JSON records trajectory eligibility as 85,542 eligible and
52,593 excluded across 138,135 entity-horizon rows. It does not record the
eligible trajectory count separately at horizons 5, 10, and 15; those three
values are therefore not asserted here. In particular, the horizon-15 cohort
must not be treated as representative of the full source cohort.

## 6. Exclusion interpretation

- C07 contributes 38 ledger rows: 35 descriptive target-frame exclusions and
  three distinct future-separation horizon-sample exclusions at horizons 5,
  10, and 15.
- C13 contributes 71,814 ledger rows and is the dominant exclusion reason. It
  represents insufficient horizon-label availability at the applicable
  entity- or target-horizon unit.
- All other primary reason counts are zero.
- Longer horizons form progressively smaller cohorts because fewer supplied
  trajectories and defender sets remain evaluable for the complete horizon.

## 7. Validation evidence

- Aggregate table reconciliation: **PASS**
- Chronological, game-level split validation: **PASS**
- Duplicate frozen-table keys: 0
- Duplicate exclusion-ledger IDs: 0
- Milestone 1 reference totals reproduced: 4,880,579 input rows, 562,936
  output rows, 272 games, 14,108 source plays, 396,914 descriptive target
  frames before exclusion, and 35 zero-defender frame exclusions

## 8. Artifact inventory

The atomic artifact set contains seven Parquet tables:

- `source_plays.parquet`
- `descriptive_target_frames.parquet`
- `primary_origins.parquet`
- `trajectory_eligibility.parquet`
- `future_separation_eligibility.parquet`
- `pair_exclusions.parquet`
- `exclusion_ledger.parquet`

It also contains `cohort_summary.json` and `manifest.json`. Files were staged
and validated before destination replacement. The manifest records relative
paths, ordered schemas and keys, row counts, file sizes, and SHA-256 checksums;
the Parquet tables were read back before publication.

## 9. Claim boundary

These cohorts support target-centric analysis of the competition-designated
receiver and the observed defenders, including scoped separation, closing, and
supplied-trajectory analysis. They do not support:

- complete passing-window claims;
- quarterback target-selection analysis;
- full-field defensive-control claims; or
- official coverage assignments.

Nearest-defender relationships remain geometric and must not be interpreted as
official coverage responsibility.

## 10. Decision

The cohort pipeline is approved for **Milestone 2 Task 2: reversible coordinate
normalization**.
