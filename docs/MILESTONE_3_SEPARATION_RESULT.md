# Milestone 3 Full-Season Separation Result

## 1. Status and decision

The full-season target–defender separation analysis is complete.

- Validation status: **PASS**
- Processed weeks: all 18, `2023_w01` through `2023_w18`
- Eligibility mismatches: **0**
- Decision: **GO TO BASELINE FEATURE AND PREDICTION DESIGN**

The descriptive result supports continued research. It does not yet establish
predictive value or demonstrate that a useful model can be built.

## 2. Analysis population

The validated population contains:

- 272 games;
- 14,107 plays with valid defender pairs;
- 94,293 origin target–defender pairs; and
- 61,156 future-evaluable horizon pairs.

| Horizon | Future-evaluable pairs | Plays |
|---:|---:|---:|
| H5 | 31,937 | 12,966 |
| H10 | 20,680 | 7,404 |
| H15 | 8,539 | 2,730 |

Split-level pair and play counts are:

| Split | Horizon | Pairs | Plays |
|---|---:|---:|---:|
| Development train | H5 | 21,216 | 8,648 |
| Development train | H10 | 13,488 | 4,837 |
| Development train | H15 | 5,320 | 1,722 |
| Validation | H5 | 5,211 | 2,085 |
| Validation | H10 | 3,499 | 1,245 |
| Validation | H15 | 1,620 | 502 |
| Frozen test | H5 | 5,510 | 2,233 |
| Frozen test | H10 | 3,693 | 1,322 |
| Frozen test | H15 | 1,599 | 506 |

The source cohort contains 14,108 plays. The one source play without valid
defender pairs is consistent with the previously documented C07 zero-defender
exclusion; it is not compared directly with the smaller future-evaluable
populations.

## 3. Definitions

The fixed outcome is:

```text
separation_change = separation_future - separation_origin
```

A negative value means contracting separation, zero means unchanged
separation, and a positive value means expanding separation.

Pairs are conditioned on the competition-designated target. The primary
analysis includes every valid observed defender represented in the applicable
future-evaluable cohort. The nearest observed defender is a secondary
descriptive reduction only. No defender is identified as an official coverage
assignment.

## 4. Primary all-pair result

Pair-weighted full-release estimates are:

| Horizon | Mean change | Median change | Closing fraction |
|---:|---:|---:|---:|
| H5 | -0.234 | -0.238 | 0.570 |
| H10 | -0.173 | -0.198 | 0.535 |
| H15 | 0.424 | 0.078 | 0.488 |

The JSON reports play-weighted uncertainty separately within each predetermined
split and horizon:

| Split | H | Pair mean / median | Pair closing | Play-weighted mean [95% interval] | Play-weighted closing [95% interval] |
|---|---:|---:|---:|---:|---:|
| Development train | 5 | -0.266 / -0.257 | 0.576 | -0.338 [-0.359, -0.312] | 0.602 [0.595, 0.609] |
| Development train | 10 | -0.238 / -0.223 | 0.540 | -0.282 [-0.342, -0.226] | 0.558 [0.549, 0.567] |
| Development train | 15 | 0.362 / 0.025 | 0.497 | 0.275 [0.140, 0.408] | 0.510 [0.498, 0.525] |
| Validation | 5 | -0.166 / -0.189 | 0.552 | -0.269 [-0.319, -0.218] | 0.587 [0.573, 0.603] |
| Validation | 10 | -0.053 / -0.119 | 0.518 | -0.184 [-0.296, -0.075] | 0.545 [0.526, 0.562] |
| Validation | 15 | 0.499 / 0.150 | 0.476 | 0.365 [0.112, 0.593] | 0.496 [0.473, 0.521] |
| Frozen test | 5 | -0.174 / -0.208 | 0.564 | -0.270 [-0.323, -0.220] | 0.595 [0.580, 0.609] |
| Frozen test | 10 | -0.050 / -0.183 | 0.532 | -0.149 [-0.265, -0.034] | 0.554 [0.535, 0.571] |
| Frozen test | 15 | 0.555 / 0.152 | 0.470 | 0.426 [0.193, 0.680] | 0.480 [0.454, 0.506] |

H5 contraction appeared in all 18 weeks and all three splits. H10 contraction
was smaller: mean change was negative in 16 of 18 weeks, while the closing
fraction exceeded one-half in all weeks and splits. H15 generally showed
expansion and used a much smaller, selective cohort.

## 5. Horizon interpretation

Within the observed defensive geometry, separation around the target tended to
contract over the first 5–10 supplied future frames. At 15 frames, separation
generally expanded within the remaining evaluable cohort.

The horizon cohorts differ and shrink substantially from H5 to H15. These
estimates are not repeated measurements on one universal set of source plays.
They therefore do not establish that defenders first close and then lose
separation on the same population.

## 6. Origin-separation findings

The following cells report pair-weighted mean change / closing fraction:

| Origin separation | H5 | H10 | H15 |
|---|---:|---:|---:|
| `[0,3)` | 0.277 / 0.451 | 0.564 / 0.448 | 0.783 / 0.454 |
| `[3,5)` | 0.242 / 0.500 | 1.104 / 0.442 | 2.618 / 0.403 |
| `[5,10)` | -0.133 / 0.555 | 0.709 / 0.448 | 2.624 / 0.333 |
| `[10,15)` | -0.843 / 0.679 | -1.201 / 0.635 | 0.070 / 0.509 |
| `[15,20)` | -1.303 / 0.783 | -3.050 / 0.805 | -3.234 / 0.731 |
| `[20,+inf)` | -1.391 / 0.803 | -3.599 / 0.866 | -5.723 / 0.885 |

Pairs initially below 5 source-field units generally expanded on average.
Pairs from 5–10 contracted at H5 but not consistently at longer horizons.
Pairs beginning at least 15 units away consistently contracted.

A plausible geometric interpretation is that far-away observed defensive
entities have more room to move closer. Initial separation therefore strongly
influences the all-pair aggregate. This is a descriptive association, not a
causal conclusion.

## 7. Defender-count finding

The `5+` origin-defender bucket overwhelmingly dominates the future-evaluable
population: 31,633 H5 pairs, 20,433 H10 pairs, and 8,419 H15 pairs. The `3–4`
bucket is small, with 304, 247, and 120 pairs respectively. No applicable
future-evaluable pairs occurred in the `1–2` bucket.

This confirms that the primary result describes broad observed defensive
geometry rather than isolated one-on-one assignments. Defender count is not
the number of defenders assigned to the target.

## 8. Nearest observed defender analysis

**nearest observed defender at origin — descriptive only, not a coverage assignment**

| Horizon | Mean change | Median change | Closing fraction |
|---:|---:|---:|---:|
| H5 | -0.038 | -0.089 | 0.545 |
| H10 | 0.122 | -0.070 | 0.526 |
| H15 | 0.563 | 0.041 | 0.486 |

H5 average change was approximately flat; H10 and H15 average separation
expanded. This nearest-observed-defender result is weaker than the all-pair
contraction result. It limits any claim that a presumed primary defender
consistently closes on the target.

## 9. Validation evidence

- Eligibility mismatches: 0
- Duplicate pair keys: 0
- Unmatched required trajectories: 0; expected and constructed eligible
  horizon-pair keys reconciled exactly
- Non-finite distances: 0
- All 18 weeks processed exactly once and in chronological order
- Aggregate horizon-pair count reconciled to the summed weekly count: 61,156
- Play-cluster bootstrap: deterministic, seed 2026, 500 resamples
- Validation status: **PASS**

## 10. Claim boundary

This result does not establish:

- official coverage assignments;
- quarterback target-selection behavior;
- complete passing-window openness;
- causal defensive effects;
- predictive performance; or
- betting or decision-making value.

Multiple target–defender pairs from the same play are dependent. Split-level
uncertainty was therefore calculated by resampling plays, not individual
pairs.

## 11. Decision and next analytical task

The descriptive association is sufficiently stable to justify a narrowly
scoped baseline prediction study. The next task is to specify, without
implementing:

- leakage-safe origin-only features;
- horizon-specific targets;
- simple non-ML and linear baselines;
- preserved development, validation, and frozen-test separation; and
- a model-selection rule that does not inspect frozen-test performance.

This report does not implement feature engineering, baseline models, or model
selection.
