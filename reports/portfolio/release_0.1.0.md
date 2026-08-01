# Gridiron Spatial Intelligence — Data-Free Release Demo

## 1. Release identity

- Project: **Gridiron Spatial Intelligence**
- Distribution: `gridiron-spatial-intelligence`
- Version: `0.1.0`
- Manifest format: `release_evidence_manifest_v1`
- Evidence files: `9`
- Evidence validation: **PASS**

## 2. Research question

> How origin geometry for a competition-designated target and observed defensive entities relates to future target–defender separation dynamics.

The original passing-window concept was narrowed because the available data lack complete 22-player tracking, all receiving options, and direct ball-path information.

## 3. Pipeline

1. Source and schema audit
2. Deterministic analytic cohort and exclusion ledger
3. Reversible coordinate normalization
4. Target–defender pair construction
5. Full-season descriptive separation analysis
6. Chronological development/validation model selection
7. One-time frozen evaluation
8. Interpretation, calibration, and validation-error diagnostics
9. Checksum-backed compact release evidence

## 4. Scale

| Quantity | Count |
|---|---:|
| Games | 272 |
| Source plays | 14,108 |
| Normalized entity-frame rows | 5,443,515 |
| Origin target–defender pairs | 94,293 |
| Horizon-evaluable pairs | 61,156 |
| H5 pairs | 31,937 |
| H10 pairs | 20,680 |
| H15 pairs | 8,539 |

## 5. Descriptive separation

| Horizon | All-pair mean change | Nearest-origin-defender mean change |
|---:|---:|---:|
| H5 | -0.234 | -0.038 |
| H10 | -0.173 | 0.122 |
| H15 | 0.424 | 0.563 |

Negative change means contraction; positive change means expansion. All-pair separation contracted most at H5, contracted less at H10, and expanded in the smaller H15 cohort. The nearest-origin-defender pattern was weaker and less consistent. These are aggregate patterns, not universal trajectories, and nearest does not mean official coverage.

## 6. Frozen predictive result

Differences are selected minus comparator; negative values favor the selected model.

### Regression

| Horizon | Selected MAE | Comparator MAE | Difference |
|---:|---:|---:|---:|
| H5 | 1.222541 | 1.299371 | -0.076830 |
| H10 | 2.424684 | 2.635341 | -0.210657 |
| H15 | 3.388624 | 3.920724 | -0.532100 |

### Classification

| Horizon | Selected log loss | Comparator log loss | Difference |
|---:|---:|---:|---:|
| H5 | 0.631936 | 0.667316 | -0.035380 |
| H10 | 0.620705 | 0.659148 | -0.038443 |
| H15 | 0.611150 | 0.661968 | -0.050818 |

- Validation-to-frozen direction agreement: `12/12`
- Frozen reversals: `0`
- Frozen evaluator executions: `1`
- Selections changed: `0`
- Comparators changed: `0`

## 7. Robustness

- Milestone 4: **GO — ORIGIN GEOMETRY HAS REPRODUCIBLE OUT-OF-SAMPLE PREDICTIVE SIGNAL**
- Milestone 5: **LIMITED ROBUSTNESS — PROCEED WITH CAUTION**

Separation-only was the weakest feature ablation; defender context mattered more at H10/H15, while absolute field location contributed little. Validation probabilities were somewhat too extreme, with H10 showing the best calibration. Performance was strongest below 15 yards. Long-separation buckets showed comparator reversals with limited support, and every H15 error bucket had limited support.

## 8. Leakage controls

- Weeks 01–12: development fitting.
- Weeks 13–15: validation selection and diagnostics.
- Weeks 16–18: evaluated exactly once.
- Validation rows used for final fitting: `0`.
- Cross-split play overlap: `0`.
- Bootstrap unit: `play`.
- The frozen result is preserved through exact-byte checksums.
- Interpretation, calibration, and error diagnostics each accessed `0` frozen-test weeks.

## 9. Release evidence

| Relative path | Role | SHA-256 | Bytes |
|---|---|---|---:|
| `artifacts/milestone_2/cohorts/cohort_summary.json` | cohort validation summary | `d88c2366a101ae79b2695b4663cf8ba0b804eb3e33a6dcbc43051ffbf4c1de53` | 5,032 |
| `artifacts/milestone_2/cohorts/manifest.json` | cohort artifact manifest | `10accd0b47a057836700cc6967f2ce5e421ac232944364f28f33e5d452e5a3c4` | 12,297 |
| `artifacts/milestone_2/normalized_tracking/manifest.json` | normalized tracking artifact manifest | `fc7dd76b03e775880bdcd896cd2c052dbf564cdb1fe37060ce5674ea487dcba4` | 17,875 |
| `artifacts/milestone_3/full_season_separation_summary.json` | full-season separation summary | `4f0fd2474636924377e793ebe4df0b8340b0c87a5b3775dca1c3dd05db4bb718` | 89,515 |
| `artifacts/milestone_4/baseline_selection.json` | audited baseline selection | `5cea3a52a368a3517274e16229e285f359c0fb56c78e8c05bd641b269e6efa17` | 101,049 |
| `artifacts/milestone_4/frozen_test_result.json` | one-time frozen-test result | `ade2fa073f8cfdd859cac41ac112be560a2a2d90872c24f71f00c2532d3cf28e` | 48,745 |
| `artifacts/milestone_5/model_interpretation_summary.json` | model interpretation summary | `4134641302848bf9be0371b3928ccb247488d548980dcb8b3a33e551f050d39e` | 63,023 |
| `artifacts/milestone_5/classifier_calibration_summary.json` | classifier calibration summary | `975ff14a46e26e85afdfac5986c9be04899a7e68a9e9072729452953ecf4c5a6` | 33,782 |
| `artifacts/milestone_5/validation_error_summary.json` | validation error summary | `3e538615481448234a8aaa87d695b439685e03b6a5801ff3d8de23e2195d0e08` | 90,627 |

- Aggregate evidence bytes: `461,945`
- Manifest validation: **PASS**

## 10. Claim boundary

This project does not establish:

- official coverage responsibility;
- quarterback decision quality or target selection;
- completion probability;
- causal receiver or defender effects;
- complete passing-window openness;
- calibration on another season;
- betting value;
- production readiness.

## 11. How to verify

```bash
python -m pytest -q \
  tests/test_packaging.py \
  tests/test_release_evidence.py \
  tests/test_portfolio_demo.py

python scripts/build_release_evidence_manifest.py \
  --output artifacts/release/v0.1.0/evidence_manifest.json

python scripts/run_portfolio_demo.py --check
```

These commands require no NFL data.
