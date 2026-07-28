# Gridiron Spatial Intelligence

A spatiotemporal research project for target-centric receiver/defender spatial
dynamics, including separation, defender closing behavior, and pre-release to
post-release trajectories in NFL player-tracking data.

## Status

Milestone 1 data validation is complete with a **LIMITED GO** decision. The
local NFL Big Data Bowl 2026 Analytics release is structurally reliable for a
narrow target-centric study, but no tracking data is committed to this
repository.

The narrowed question is: for the competition-designated target and observed
coverage defenders, how do separation and defender closing patterns evolve
from the pre-release input sequence into the supplied post-release trajectory?

The current release does not support defensible claims about all receiving
options, complete defensive space control, quarterback target selection,
full-field passing windows, or direct ball-path obstruction. The term
“passing window” is reserved for a later dataset or extension with sufficient
full-field and ball-path context.

See [the project plan](docs/PROJECT_PLAN.md) for the data requirements,
leakage-safe evaluation design, baseline-first roadmap, visualization concept,
and MVP scope. See [the Milestone 1 result](docs/MILESTONE_1_RESULT.md) for the
validated claim boundary.

## Repository layout

- `docs/` — research plan and, after data acquisition, the data audit.
- `data/raw/` — authorized source data (ignored by Git).
- `data/processed/` — reproducible derived data (ignored by Git).
- `notebooks/` — milestone-oriented exploratory analysis.
- `src/gridiron_spatial/` — reusable research code introduced only as needed.
- `artifacts/` — generated figures, reports, and manifests (ignored by Git).
- `tests/` — checks for geometry, labels, and leakage boundaries.

## Current next step

The single recommended Milestone 2 task is to write the restricted analytic
cohort and coordinate-contract specification before creating features or
models. That task has not been implemented.
