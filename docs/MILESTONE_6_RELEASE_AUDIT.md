# Milestone 6 Portfolio-Release Audit

## 1. Current project identity

| Item | Current state |
|---|---|
| Package | `gridiron-spatial-intelligence` |
| Package version | `0.1.0` |
| Python requirement | Python `>=3.11` |
| Declared runtime dependencies | NumPy, pandas, PyArrow, and scikit-learn |
| README purpose | An NFL tracking-data research portfolio focused on target-centric receiver/defender spatial dynamics |
| Primary research claim | Origin geometry has reproducible average predictive signal for future target–defender separation dynamics, with meaningful geometric heterogeneity |
| Milestone 4 decision | `GO — ORIGIN GEOMETRY HAS REPRODUCIBLE OUT-OF-SAMPLE PREDICTIVE SIGNAL` |
| Milestone 5 decision | `LIMITED ROBUSTNESS — PROCEED WITH CAUTION` |

The repository presents a bounded research baseline, not a complete passing-window system, official coverage model, or production service.

## 2. Verified accomplishments

| Stage | Tracked source | Entry script | Focused tests | Tracked report or contract |
|---|---|---|---|---|
| 1. Dataset and source audit | `src/gridiron_spatial/data_audit.py`, `milestone_1_validation.py` | `scripts/run_data_audit.py`, `run_milestone_1_validation.py` | `test_data_audit.py`, `test_milestone_1_validation.py` | `docs/data_audit.md`, `MILESTONE_1.md`, `MILESTONE_1_RESULT.md` |
| 2. Analytic cohort construction | `cohort.py`, `cohort_artifacts.py` | `build_cohort_artifacts.py`, cohort smoke scripts | `test_cohort.py`, `test_cohort_artifacts.py`, `test_build_cohort_artifacts.py`, `test_smoke_two_week_cohort.py` | `ANALYTIC_COHORT_AND_COORDINATE_CONTRACT.md`, `MILESTONE_2.md`, `MILESTONE_2_COHORT_RESULT.md` |
| 3. Reversible coordinate normalization | `coordinates.py`, `coordinate_frame.py`, `normalized_tracking.py`, `normalized_artifacts.py` | `build_normalized_tracking.py`, `smoke_week_normalization.py` | `test_coordinates.py`, `test_coordinate_frame.py`, `test_normalized_tracking.py`, `test_normalized_artifacts.py`, `test_build_normalized_tracking.py` | Coordinate contract in `ANALYTIC_COHORT_AND_COORDINATE_CONTRACT.md` |
| 4. Target–defender pair construction | `receiver_defender_pairs.py` | `smoke_week_receiver_defender_pairs.py` | `test_receiver_defender_pairs.py` | `MILESTONE_3_SEPARATION_RESULT.md` |
| 5. Full-season descriptive analysis | `separation_summary.py` | `analyze_full_season_separation.py` | `test_separation_summary.py` | `MILESTONE_3_SEPARATION_RESULT.md` |
| 6. Leakage-safe temporal model selection | `baseline_features.py`, `baseline_models.py` | `select_baseline_models.py` | `test_baseline_features.py`, `test_baseline_models.py` | `MILESTONE_4_BASELINE_PLAN.md`, `MILESTONE_4_SELECTION_AUDIT.md` |
| 7. One-time frozen evaluation | `frozen_evaluation.py` | `evaluate_frozen_baselines.py` | `test_frozen_evaluation.py` | `MILESTONE_4_BASELINE_RESULT.md` |
| 8. Coefficient stability and ablations | `model_interpretation.py` | `analyze_model_interpretation.py` | `test_model_interpretation.py` | `MILESTONE_5_INTERPRETATION_RESULT.md` |
| 9. Classifier calibration | `calibration_analysis.py` | `analyze_classifier_calibration.py` | `test_calibration_analysis.py` | `MILESTONE_5_INTERPRETATION_RESULT.md` |
| 10. Validation error analysis | `error_analysis.py` | `analyze_validation_errors.py` | `test_error_analysis.py` | `MILESTONE_5_INTERPRETATION_RESULT.md` |

Recent history records each late-stage result separately, culminating in commit `3c444e1` for model interpretation and robustness.

## 3. Reproducibility entry points

| Workflow | Existing entry point | Release treatment |
|---|---|---|
| Cohort construction | `python scripts/build_cohort_artifacts.py ...` | Document required local release paths and explicit week list |
| Normalized tracking | `python scripts/build_normalized_tracking.py ...` | Document as an optional full-data rebuild |
| Full-season separation analysis | `python scripts/analyze_full_season_separation.py ...` | Document as an optional full-data analysis |
| Development/validation selection | `python scripts/select_baseline_models.py ...` | Reproducible only with authorized Weeks 01–15 artifacts |
| Frozen evaluation | `python scripts/evaluate_frozen_baselines.py ...` | **Do not advertise as rerunnable** |
| Coefficient stability and ablations | `python scripts/analyze_model_interpretation.py ...` | Weeks 01–15 only |
| Calibration | `python scripts/analyze_classifier_calibration.py ...` | Weeks 01–15 only |
| Error analysis | `python scripts/analyze_validation_errors.py ...` | Weeks 01–15 only |

The frozen evaluator is historical protocol, not a general command. Release documentation should publish its immutable aggregate result and checksum, state that execution count was one, and explicitly instruct readers not to rerun it. Reproduction should stop at development/validation workflows unless a wholly new study preregisters a new test set.

The current repository has real-data smoke scripts, but no tracked data-free portfolio demonstration that consumes only compact release summaries.

## 4. Test and quality inventory

- Tracked test modules: `19`.
- Authorized collection result: `78` tests were enumerated before collection stopped with `5` module import errors.
- Affected modules: `test_baseline_models.py`, `test_calibration_analysis.py`, `test_error_analysis.py`, `test_frozen_evaluation.py`, and `test_model_interpretation.py`.
- Exact collection failure: `ModuleNotFoundError: No module named 'sklearn'` in the active `python` interpreter.
- Full collected-test count is therefore not established by this audit.
- No tests were executed, and this audit makes no test-pass claim.
- Linting: no dedicated tracked lint command or configuration is evident.
- Static type checking: no dedicated tracked type-check command or configuration is evident.
- CI: absent; `.github/` does not exist.
- Package import coverage: package imports occur throughout tests, but no explicit clean-environment import smoke is tracked.
- CLI checks: synthetic CLI/argument tests exist for cohort and normalized builders and week parsing.
- Smoke checks: several real-data smoke scripts exist; they are not lightweight release checks.
- Release verification: not automated.

The failed collection is an environment/reproducibility finding, not evidence that the tests themselves fail.

## 5. Documentation inventory

| Documentation element | Classification | Evidence |
|---|---|---|
| README quickstart | Present but incomplete | `README.md` exists, but there is no tracked data-free demo or release-focused command path |
| Project architecture | Missing | No tracked architecture document |
| Data contract | Present and sufficient | `ANALYTIC_COHORT_AND_COORDINATE_CONTRACT.md` |
| Model card | Missing | No tracked model-card file |
| Evaluation methodology | Present and sufficient | Milestone 4 plan, selection audit, and result |
| Limitations and ethics | Present but incomplete | Strong claim boundaries exist in milestone results, but are fragmented |
| Reproducibility guide | Missing | Commands exist in scripts but are not consolidated |
| Release checklist | Missing | No tracked release checklist |
| Interview/portfolio notes | Missing | No reviewer-oriented summary |
| Changelog | Missing | No `CHANGELOG.md` |
| Citation file | Missing | No `CITATION.cff` |
| License | Missing | No tracked license |
| Contributing guide | Missing | No `CONTRIBUTING.md` |

## 6. Artifact policy

Observed policy:

- `git ls-files` contains only `artifacts/.gitkeep`; generated JSON, manifests, and Parquet outputs are untracked.
- `data/raw/` and `data/processed/` contain only tracked `.gitkeep` placeholders, so large/raw NFL data is excluded from Git.
- The clean pre-audit status while local generated artifacts exist confirms that generated outputs are ignored.
- Existing builders use manifests, checksums, atomic writes, and project-relative output references.
- Checksum and manifest behavior is described in milestone documentation, but no versioned release manifest preserves the final compact-result checksums.
- The frozen-test report is tracked, but its machine-readable result is not. Its immutability policy is described narratively rather than enforced by a release checksum.

Recommended explicit policy:

1. Never track raw CSV/ZIP data, normalized tracking Parquet, cohort Parquet, pair rows, predictions, fitted models, or caches.
2. Continue regenerating large derived artifacts locally.
3. Track an allowlisted set of compact aggregate release JSON files, or immutable copies under `results/v0.1.0/`.
4. Track a release manifest containing relative paths, schemas, SHA-256 checksums, source commit, and frozen-evaluator execution count.
5. Mark the frozen result immutable and never regenerate it for release hardening.
6. Reject absolute local paths from every tracked result.

## 7. Portfolio credibility

A reviewer can determine the narrowed research question, completed analytical stages, strongest result, and major limitations by reading several milestone reports. Leakage prevention is particularly well documented in the cohort contract and Milestone 4 materials.

The two-minute portfolio experience is weaker:

- the README does not yet act as a concise landing page for the final result and narrowed scope;
- architecture and reproduction steps are distributed across milestone documents;
- there is no data-free demonstration;
- model limitations are not consolidated in a model card;
- aggregate machine-readable evidence is not tracked; and
- CI provides no visible verification signal.

The highest-impact improvements are a result-first README, a data-free summary demo, tracked compact release evidence, a concise architecture/reproducibility guide, and a green CI release gate.

## 8. Release blockers

| Severity | Affected file or missing file | Evidence | Smallest corrective action | Data processing? |
|---|---|---|---|---|
| `BLOCKER` | `pyproject.toml`, `README.md` install instructions | Authorized collection stopped with five `sklearn` import errors | Verify dependency declaration, document a clean virtual-environment install, and make the documented collection command succeed | No |
| `BLOCKER` | `.gitignore`, untracked `artifacts/milestone_4/*.json` and `artifacts/milestone_5/*.json`, missing release manifest | Only `artifacts/.gitkeep` is tracked; frozen evidence has no versioned checksum | Allowlist compact aggregate results and add an immutable checksum manifest without rerunning analyses | No |
| `IMPORTANT` | `README.md`, missing lightweight demo | No data-free reviewer path exists | Add a result-first quickstart and a summary-only demo | No |
| `IMPORTANT` | Missing `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md`, `docs/MODEL_CARD.md` | Required information is spread across milestone documents | Consolidate architecture, exact commands, model limits, and claim boundaries | No |
| `IMPORTANT` | Missing `.github/workflows/ci.yml` | `.github/` is absent | Add install, collection/test, compilation, and data-free smoke jobs | No |
| `IMPORTANT` | `scripts/evaluate_frozen_baselines.py`, release documentation | A one-time evaluator exists alongside rerunnable scripts | Label it historical/non-rerunnable and verify its stored checksum instead | No |
| `IMPORTANT` | Missing `LICENSE`, `CHANGELOG.md`, `CITATION.cff`, release checklist | None are tracked | Add release-governance files after owner selects a license | No |
| `OPTIONAL` | Missing `CONTRIBUTING.md` | No contributor workflow | Add a short contribution and data-boundary guide | No |
| `OPTIONAL` | Missing portfolio/interview notes | No two-minute reviewer narrative beyond README | Add a one-page project walkthrough if useful | No |

Counts: `2 BLOCKER`, `5 IMPORTANT`, `2 OPTIONAL`.

These blockers prevent tagging today, but they are release-hardening gaps rather than defects in the completed research design.

## 9. Proposed release scope

### `v0.1.0 — validated research baseline and reproducible portfolio release`

Include:

- the Python package, scripts, and synthetic tests;
- the analytic cohort and coordinate contracts;
- milestone result reports and claim boundaries;
- allowlisted compact aggregate result JSON plus checksum manifest;
- a result-first README, architecture overview, reproducibility guide, and model card;
- one data-free demonstration based only on tracked aggregate results;
- CI and a release checklist; and
- license, changelog, and citation metadata.

Explicitly exclude:

- raw or derived NFL row-level data;
- automatic downloading of restricted data;
- new model families or tuning;
- new frozen-test selection or rerunning the frozen evaluator;
- predictions, fitted pipelines, and full Parquet artifacts;
- production-deployment claims; and
- betting claims.

## 10. Ordered implementation plan

| Order | Independently committable task | Expected files | Validation | Scope | Use Codex? |
|---:|---|---|---|---|---|
| 1 | Stabilize clean-environment installation and collection | `pyproject.toml`, `README.md` | clean venv install, `python -m pip check`, `python -m pytest --collect-only -q` | Small | Yes |
| 2 | Version compact evidence and freeze release checksums | `.gitignore`, allowlisted `results/v0.1.0/*.json`, `results/v0.1.0/manifest.json`, `docs/REPRODUCIBILITY.md` | schema/checksum unit test; verify frozen checksum without evaluation | Medium | Yes |
| 3 | Build the portfolio documentation layer | `README.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/REPRODUCIBILITY.md` | link/command review; `git diff --check` | Medium | Yes |
| 4 | Add one aggregate-only demonstration | `scripts/demo_release_summary.py`, `tests/test_demo_release_summary.py` | focused synthetic/fixture test and CLI smoke with no NFL data | Small | Yes |
| 5 | Add release governance and CI | `.github/workflows/ci.yml`, `LICENSE`, `CHANGELOG.md`, `CITATION.cff`, `docs/RELEASE_CHECKLIST.md` | CI green; release-checklist dry run | Medium | Yes, after owner chooses the license |

The plan ends in a release gate and does not expand predictive research.

## 11. Recommended release gate

Before tagging:

1. Create a clean supported-Python virtual environment and install the package with development dependencies.
2. Run `python -m pip check`.
3. Run complete pytest collection and the full pytest suite.
4. Run `python -m compileall -q src scripts`.
5. Run configured lint and type checks.
6. Build and install the package artifact, then verify a clean package import.
7. Execute every README quickstart command.
8. Run the data-free demonstration smoke check.
9. Confirm CI is green on the release commit.
10. Verify that Git tracks no raw CSV/ZIP, NFL-derived Parquet, predictions, fitted models, or caches.
11. Verify every allowlisted compact result against the release manifest.
12. Verify the frozen-result checksum and recorded one-time evaluator count without rerunning it.
13. Run `git diff --check` and require a clean `git status --short`.
14. Confirm the version, changelog, citation, license, model card, and release checklist agree.

## 12. Final audit decision

**READY FOR BOUNDED RELEASE HARDENING**

The scientific baseline, leakage controls, frozen evaluation, and robustness analysis are sufficiently complete for a bounded portfolio release. The repository should not be tagged until clean-environment test collection and versioned compact-result provenance are corrected. No further predictive research is required for `v0.1.0`.
