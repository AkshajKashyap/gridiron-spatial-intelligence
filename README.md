# Gridiron Spatial Intelligence

Target-centric NFL tracking research on how origin geometry relates to future
receiver–defender separation dynamics.

- Release: **v0.1.0**
- Distribution: `gridiron-spatial-intelligence`
- Milestone 4: **GO — origin geometry has reproducible out-of-sample predictive signal**
- Milestone 5: **LIMITED ROBUSTNESS — PROCEED WITH CAUTION**

## Why this project exists

The original motivation was to study spatial passing opportunities: how
receivers, defenders, and trajectories shape the space available for a throw.
The source audit forced a narrower and more defensible question. The
competition release does not provide complete 22-player tracking, trajectories
for every receiving option, or direct ball-path information.

The binding research question is:

> How origin geometry for a competition-designated target and observed
> defensive entities relates to future target–defender separation dynamics.

The target is supplied by the competition data; this project does not predict
the quarterback's choice. “Nearest defender” is a deterministic geometric
relationship, not an official coverage assignment.

## What was built

The repository implements and tests an end-to-end research pipeline:

1. source-file, schema, key, coordinate, and timing audit;
2. deterministic analytic cohorts with an exclusion ledger;
3. reversible left/right coordinate normalization;
4. target–defender pairing at an immutable prediction origin;
5. full-season descriptive separation analysis;
6. leakage-safe selection among preregistered linear and logistic baselines;
7. one-time evaluation on a frozen chronological test split;
8. coefficient-stability and fixed feature-group ablations;
9. classifier calibration diagnostics;
10. validation errors sliced by fixed origin-separation buckets; and
11. a checksum-backed manifest for compact release evidence.

## Key scale and results

| Evidence | Value |
|---|---:|
| Games | 272 |
| Source plays | 14,108 |
| Normalized entity-frame rows | 5,443,515 |
| Origin target–defender pairs | 94,293 |
| Future-evaluable horizon pairs | 61,156 |
| Validation-to-frozen primary directions agreeing | 12 / 12 |
| Frozen direction reversals | 0 |

Origin geometry improved frozen out-of-sample performance over the registered
constant or origin-separation-only comparators. Differences below are
selected-model score minus comparator score, so negative values favor the
selected model.

| Frozen population/task | Horizon | Selected | Comparator | Difference |
|---|---:|---:|---:|---:|
| All-pair regression MAE | H5 | 1.222541 | 1.299371 | -0.076830 |
| All-pair regression MAE | H10 | 2.424684 | 2.635341 | -0.210657 |
| All-pair classification log loss | H10 | 0.620705 | 0.659148 | -0.038443 |
| Nearest-defender regression MAE | H15 | 1.594830 | 1.602184 | -0.007354 |

The last row is deliberately included as a weak result: its play-clustered
interval crossed zero. More broadly, performance varied by origin-separation
regime, longer-horizon cohorts were progressively smaller, and validation
probabilities were somewhat overconfident. This is reproducible predictive
signal, not uniform reliability or demonstrated operational value.

## Research design

- **Weeks 01–12:** development fitting.
- **Weeks 13–15:** validation selection and later robustness diagnostics.
- **Weeks 16–18:** one-time frozen evaluation; not reused for selection.
- **Horizons:** H5, H10, and H15 are separate tasks with distinct eligible
  samples.
- **Regression target:** future separation minus origin separation.
- **Classification target:** `1` when separation contracts and `0` otherwise.
- **Uncertainty:** deterministic play-cluster bootstrap retains all pair rows
  from a sampled play.
- **Leakage controls:** only origin-available geometry enters features;
  future coordinates and label-availability fields are prohibited; all
  preprocessing is fit on development data.

## Data-free quickstart

These commands inspect the packaged code and compact release evidence without
NFL data:

```bash
git clone https://github.com/AkshajKashyap/gridiron-spatial-intelligence.git
cd gridiron-spatial-intelligence

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[test]"

python -m pytest -q tests/test_packaging.py tests/test_release_evidence.py

python scripts/build_release_evidence_manifest.py \
  --output artifacts/release/v0.1.0/evidence_manifest.json
```

With unchanged source evidence, rebuilding the manifest produces identical
bytes. This workflow verifies aggregate results and provenance; it does not
run the historical frozen evaluator.

## Data-free portfolio demo

```bash
python scripts/run_portfolio_demo.py
python scripts/run_portfolio_demo.py --check
```

The demo reads only tracked aggregate evidence. Its deterministic output is
versioned at
[reports/portfolio/release_0.1.0.md](reports/portfolio/release_0.1.0.md).

## Full-data requirements

NFL competition data is not included. Raw files and derived Parquet tables are
intentionally excluded from Git. Full analytical reproduction requires a user
to obtain the authorized competition release, accept its terms, and place it
locally. The repository does not automatically download or redistribute
restricted data. See [Reproducibility](docs/REPRODUCIBILITY.md).

## Repository map

| Path | Purpose |
|---|---|
| `src/gridiron_spatial/` | Cohort, geometry, feature, model, evaluation, and diagnostic logic |
| `scripts/` | Explicit pipeline and verification entry points |
| `tests/` | Synthetic, data-free unit and contract tests |
| `docs/` | Research contracts, milestone evidence, and release documentation |
| `artifacts/release/v0.1.0/` | Deterministic release-evidence manifest |
| Allowlisted milestone JSON | Compact aggregate provenance and results referenced by the manifest |

## Claim boundary and limitations

This release does **not** establish:

- official coverage responsibility;
- quarterback decision quality or target selection;
- completion probability;
- causal receiver or defender effects;
- complete passing-window openness;
- full-field defensive control;
- calibrated probabilities for another season;
- betting value; or
- production readiness.

The data represent a competition-designated target and observed defensive
entities, with future evaluation restricted to supplied output trajectories.
Only the 2023 weekly chronology is evaluated, so cross-season generalization
is unknown.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Model card](docs/MODEL_CARD.md)
- [Evaluation methodology](docs/EVALUATION_METHODOLOGY.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Interview notes](docs/INTERVIEW_NOTES.md)
- [Analytic cohort and coordinate contract](docs/ANALYTIC_COHORT_AND_COORDINATE_CONTRACT.md)
- [Milestone 4 baseline result](docs/MILESTONE_4_BASELINE_RESULT.md)
- [Milestone 5 interpretation result](docs/MILESTONE_5_INTERPRETATION_RESULT.md)
- [Release evidence manifest](artifacts/release/v0.1.0/evidence_manifest.json)

## Governance and release

- [MIT License](LICENSE)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
- [Contributing guide](CONTRIBUTING.md)
- [v0.1.0 release notes](docs/RELEASE_NOTES_0.1.0.md)
