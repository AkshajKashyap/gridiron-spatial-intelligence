# Gridiron Spatial Intelligence v0.1.0

## Release identity

- Project: Gridiron Spatial Intelligence
- Distribution: `gridiron-spatial-intelligence`
- Version: `0.1.0`
- Release date: `2026-08-01`
- Python support: 3.11 and 3.13
- Status: validated research baseline and reproducible portfolio release

## Research question

> How origin geometry for a competition-designated target and observed defensive entities relates to future target–defender separation dynamics.

## Why the scope changed

The original passing-window concept was narrowed after the source audit. The
supplied competition data does not contain complete 22-player tracking, all
receiving options, or direct ball-path information. The release therefore
focuses on defensible target-centric receiver–defender separation dynamics.

## What ships

- A tested Python package and research-pipeline source.
- 142 data-free tests.
- Research, methodology, model, and reproducibility documentation.
- An aggregate-only, data-free portfolio demo.
- Nine checksum-verified evidence files.
- Deterministic release verification.
- Python 3.11/3.13 continuous integration.

## Scale

| Quantity | Count |
|---|---:|
| Games | 272 |
| Source plays | 14,108 |
| Normalized entity-frame rows | 5,443,515 |
| Origin pairs | 94,293 |
| Horizon-evaluable pairs | 61,156 |

## Main result

All 12 selected multivariable baselines agreed in improvement direction
between validation and the one-time frozen evaluation, with zero reversals.
Representative all-pair frozen differences are:

| Task | H5 | H10 | H15 |
|---|---:|---:|---:|
| Regression MAE | -0.076830 | -0.210657 | -0.532100 |
| Classification log loss | -0.035380 | -0.038443 | -0.050818 |

These values are selected-model score minus comparator score; negative values
favor the selected model.

## Robustness limits

- Predicted probabilities were somewhat too extreme.
- Performance varied across origin-separation regimes.
- Long-separation buckets produced comparator reversals.
- H15 used a smaller, more selective cohort and should not be treated as
  broadly representative.
- The nearest observed defender is not an official coverage assignment.

## Frozen-test safeguards

| Safeguard | Result |
|---|---:|
| Execution count | 1 |
| Selections changed | 0 |
| Comparators changed | 0 |
| Leakage validation | PASS |
| Reconciliation | PASS |
| Mismatch count | 0 |

The frozen evaluation is release evidence, not rerunnable tuning
infrastructure.

## Verification

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/run_portfolio_demo.py --check
python scripts/verify_release.py
```

These commands require no NFL data.

## Data availability and claim boundary

NFL source data is not included. Full analytical reproduction requires legally
obtained competition data, and both raw data and derived Parquet files are
excluded from the release.

This release does not establish official coverage, completion probability,
causality, betting value, or production readiness.
