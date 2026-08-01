# Reproducibility

Release `v0.1.0` supports two different verification paths. The first is
data-free and checks the public release evidence. The second requires legally
obtained NFL competition data and rebuilds ignored analytical artifacts.

## 1. Data-free release verification

### Environment and installation

```bash
git clone https://github.com/AkshajKashyap/gridiron-spatial-intelligence.git
cd gridiron-spatial-intelligence

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pip check
```

The installed distribution is `gridiron-spatial-intelligence` version `0.1.0`.
Its runtime dependencies are NumPy, pandas, PyArrow, and scikit-learn.

### Focused release tests

```bash
python -m pytest -q \
  tests/test_packaging.py \
  tests/test_release_evidence.py
```

These tests check package import/metadata, the exact nine-file evidence
allowlist, JSON/path safety, SHA-256 and byte-size calculation, cross-artifact
references, frozen-policy fields, deterministic output, and atomic writing.
They do not need NFL data.

### Evidence-manifest regeneration and checksum comparison

```bash
python scripts/build_release_evidence_manifest.py \
  --output artifacts/release/v0.1.0/evidence_manifest.json

git diff --exit-code -- \
  artifacts/release/v0.1.0/evidence_manifest.json
```

The builder reads the nine compact JSON files as bytes, validates their
relationships and frozen-evaluation policy, and writes atomically. Unchanged
source evidence must produce byte-identical manifest output. A clean
`git diff` for the manifest is the direct comparison with the versioned
release evidence.

### Complete test collection

```bash
python -m pytest --collect-only -q
```

Collection verifies that every test module imports in the installed
environment. It does not execute tests or analytical pipelines.

## 2. Full analytical reproduction

### Restricted source data

Obtain the `nfl-big-data-bowl-2026-prediction` competition release through an
authorized Kaggle/NFL mechanism after accepting its terms. Place the extracted
release under:

```text
data/raw/bdb_2026/
```

The validated release contained `train/input_2023_w01.csv` through
`input_2023_w18.csv`, corresponding weekly output files, and
`supplementary_data.csv`. Source data are not shipped, scraped, or downloaded
automatically by this repository. Never commit or redistribute them.

### Pipeline order

Each entry point exposes its current CLI through `--help`. Supply explicit
chronological weeks and local input/output paths rather than relying on
implicit discovery.

1. `scripts/run_data_audit.py` — basic inventory/schema audit.
2. `scripts/run_milestone_1_validation.py` — structural validation and compact
   Milestone 1 evidence.
3. `scripts/build_cohort_artifacts.py` — six cohort tables plus exclusion
   ledger, `cohort_summary.json`, and cohort manifest.
4. `scripts/build_normalized_tracking.py` — phase-qualified normalized tracking
   partitions and normalized manifest.
5. `scripts/analyze_full_season_separation.py` — compact descriptive separation
   summary.
6. `scripts/select_baseline_models.py` — Weeks 01–15 registered selection
   artifact.
7. `scripts/analyze_model_interpretation.py` — development/validation
   coefficient stability and fixed ablations.
8. `scripts/analyze_classifier_calibration.py` — development/validation
   calibration diagnostics.
9. `scripts/analyze_validation_errors.py` — fixed validation error buckets.
10. `scripts/build_release_evidence_manifest.py` — compact-evidence integrity.

`scripts/evaluate_frozen_baselines.py` is historical one-time protocol, not a
routine reproduction step. Its final aggregate result and checksum are already
versioned. **Do not rerun it for model selection, tuning, calibration, or
release hardening.** A new study must preregister a new untouched holdout.

### Artifact policy

`build_cohort_artifacts.py` writes seven ignored Parquet tables plus compact
cohort JSON. `build_normalized_tracking.py` writes ignored weekly normalized
Parquet partitions plus a compact manifest. Pair-level rows, predictions, and
fitted objects remain ignored. The full-season, selection, frozen, and
Milestone 5 scripts write compact aggregate JSON; only the nine release-
allowlisted files are versioned.

### Reference reconciliation

A valid rebuild should reconcile these established totals:

| Quantity | Reference |
|---|---:|
| Input rows | 4,880,579 |
| Output rows | 562,936 |
| Games | 272 |
| Source plays | 14,108 |
| Descriptive target frames before C07 | 396,914 |
| Zero-defender target-frame exclusions | 35 |
| Normalized entity-frame rows | 5,443,515 |
| Origin target–defender pairs | 94,293 |
| Future-evaluable horizon pairs | 61,156 |

A mismatch must be investigated, never forced to match.

### Resource expectations

The documented full-release Milestone 1 validator took 290.998 seconds across
measured stages and peaked at 694,476 KiB (about 678 MiB) resident memory on
the original environment. Later timings are machine- and cache-dependent;
their compact evidence records measured runtimes. Budget additional space for
ignored CSV and Parquet artifacts. No production performance guarantee is
made.

## 3. Reproducibility boundary

Data-free verification establishes packaging, source-code tests, compact
evidence integrity, and provenance relationships. It does not independently
recompute research results. Full reproduction requires the legally obtained
source release and can verify the development/validation pipeline. The
versioned frozen result remains the final one-time evidence for `v0.1.0`.
