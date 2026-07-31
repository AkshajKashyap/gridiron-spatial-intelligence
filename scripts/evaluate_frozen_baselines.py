"""Execute the single authorized frozen evaluation of audited baselines."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.baseline_features import (  # noqa: E402
    HORIZONS,
    REGISTERED_NUMERIC_FEATURES,
    SAMPLE_KEY,
    build_baseline_samples,
    feature_matrix,
)
from gridiron_spatial.frozen_evaluation import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    atomic_write_json,
    evaluate_specification,
    fit_predictor,
    validate_frozen_inventory,
    validate_sample_keys,
)
from gridiron_spatial.receiver_defender_pairs import (  # noqa: E402
    build_receiver_defender_pairs,
)


DEVELOPMENT_WEEKS = tuple(f"2023_w{week:02d}" for week in range(1, 13))
FROZEN_WEEKS = tuple(f"2023_w{week:02d}" for week in range(16, 19))
EXPECTED_FROZEN_COUNTS = {
    ("all_pairs", 5): (5510, 2233),
    ("all_pairs", 10): (3693, 1322),
    ("all_pairs", 15): (1599, 506),
    ("nearest_observed_defender", 5): (2161, 2161),
    ("nearest_observed_defender", 10): (1300, 1300),
    ("nearest_observed_defender", 15): (501, 501),
}
TRACKING_COLUMNS = [
    "game_id",
    "play_id",
    "phase",
    "frame_id",
    "nfl_id",
    "week",
    "split",
    "player_side",
    "player_role",
    "x_norm",
    "y_norm",
    "normalized_coordinate_class",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must be inside project: {path}") from error


def _manifest_header(path: Path, fields: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    patterns = {
        field: re.compile(rf'^\s*"{re.escape(field)}":\s*"([^"]+)"')
        for field in fields
    }
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            for field, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    found[field] = match.group(1)
            if len(found) == len(fields):
                break
            if line_number >= 12:
                break
    if set(found) != set(fields):
        raise ValueError("Artifact manifest version fields are incomplete")
    return found


def _read_cohort(root: Path, name: str, week: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        root / f"{name}.parquet", filters=[("week", "==", week)]
    )
    if frame.empty or set(frame["week"].astype("string")) != {week}:
        raise ValueError(f"Cohort table {name} does not isolate {week}")
    return frame


def _build_week(
    normalized_root: Path,
    cohort_root: Path,
    week: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tracking = pd.read_parquet(
        normalized_root / f"normalized_{week}.parquet",
        columns=TRACKING_COLUMNS,
    )
    if set(tracking["week"].astype("string")) != {week}:
        raise ValueError(f"Normalized partition does not isolate {week}")
    origins = _read_cohort(cohort_root, "primary_origins", week)
    trajectories = _read_cohort(
        cohort_root, "trajectory_eligibility", week
    )
    future = _read_cohort(
        cohort_root, "future_separation_eligibility", week
    )
    pairs = build_receiver_defender_pairs(
        tracking, origins, trajectories, future
    )
    if pairs.diagnostics["eligibility_mismatch_count"]:
        raise ValueError(f"Pair eligibility mismatch in {week}")
    samples = build_baseline_samples(pairs.pairs, pairs.origin_pairs)
    if len(samples) != pairs.diagnostics["constructed_horizon_pairs"]:
        raise ValueError(f"Sample reconciliation mismatch in {week}")
    diagnostic = {
        "week": week,
        "expected_horizon_pairs": pairs.diagnostics[
            "expected_horizon_pairs"
        ],
        "constructed_horizon_pairs": pairs.diagnostics[
            "constructed_horizon_pairs"
        ],
        "sample_count": int(len(samples)),
        "mismatch_count": 0,
    }
    return samples, diagnostic


def _population(
    samples: pd.DataFrame,
    population: str,
    horizon: int,
) -> pd.DataFrame:
    result = samples.loc[samples["horizon"].eq(horizon)]
    if population == "nearest_observed_defender":
        result = result.loc[
            result["nearest_observed_defender_indicator"].eq(1)
        ]
    return result.copy(deep=True)


def _summary(samples: pd.DataFrame) -> dict[str, Any]:
    target = samples["separation_change"]
    return {
        "sample_count": int(len(samples)),
        "play_count": int(
            samples[["game_id", "play_id"]].drop_duplicates().shape[0]
        ),
        "separation_change_mean": float(target.mean()),
        "separation_change_median": float(target.median()),
        "closing_rate": float(samples["closing"].mean()),
    }


def _validate_samples(
    samples: pd.DataFrame,
    *,
    expected_weeks: tuple[str, ...],
    expected_split: str,
) -> None:
    validate_sample_keys(samples)
    if tuple(sorted(samples["week"].astype("string").unique())) != expected_weeks:
        raise ValueError("Sample weeks do not match authorized boundary")
    if set(samples["split"].astype("string")) != {expected_split}:
        raise ValueError("Sample split does not match authorized boundary")
    if list(feature_matrix(samples).columns) != list(
        REGISTERED_NUMERIC_FEATURES
    ):
        raise ValueError("Feature matrix differs from frozen allowlist")
    ordered = samples.sort_values(list(SAMPLE_KEY), kind="stable").reset_index(
        drop=True
    )
    if not samples.reset_index(drop=True).equals(ordered):
        raise ValueError("Samples are not deterministically ordered")


def _console(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for evaluation in result["evaluations"]:
        differences = evaluation["metric_differences"]
        intervals = evaluation["play_cluster_bootstrap_95_intervals"]
        row = {
            "population": evaluation["population"],
            "horizon": evaluation["horizon"],
            "frozen_samples": evaluation["frozen_summary"]["sample_count"],
            "frozen_plays": evaluation["frozen_summary"]["play_count"],
            "selected_candidate": evaluation["selected_candidate"],
            "comparator_candidate": evaluation["comparator_candidate"],
        }
        if evaluation["task"] == "regression":
            row.update(
                {
                    "selected_mae": evaluation["selected_metrics"]["mae"],
                    "comparator_mae": evaluation["comparator_metrics"]["mae"],
                    "mae_difference": differences["mae"],
                    "mae_difference_interval": intervals["mae_difference"],
                }
            )
        else:
            row.update(
                {
                    "selected_log_loss": evaluation["selected_metrics"][
                        "log_loss"
                    ],
                    "comparator_log_loss": evaluation[
                        "comparator_metrics"
                    ]["log_loss"],
                    "log_loss_difference": differences["log_loss"],
                    "log_loss_difference_interval": intervals[
                        "log_loss_difference"
                    ],
                    "selected_brier": evaluation["selected_metrics"][
                        "brier_score"
                    ],
                    "comparator_brier": evaluation["comparator_metrics"][
                        "brier_score"
                    ],
                    "brier_difference": differences["brier_score"],
                    "brier_difference_interval": intervals[
                        "brier_score_difference"
                    ],
                }
            )
        rows.append(row)
    return {
        "development_weeks": result["development_weeks"],
        "frozen_weeks": result["frozen_weeks"],
        "validation_rows_used_for_fitting": 0,
        "frozen_selections_reproduced": 12,
        "selection_or_comparator_changes": 0,
        "leakage_validation": result["leakage_diagnostics"]["status"],
        "reconciliation_mismatch_count": result[
            "reconciliation_diagnostics"
        ]["mismatch_count"],
        "evaluations": rows,
        "output": result["output_artifact"],
        "total_runtime_seconds": result["total_runtime_seconds"],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-weeks", nargs="+", required=True)
    parser.add_argument("--frozen-weeks", nargs="+", required=True)
    args = parser.parse_args(arguments)
    started = time.perf_counter()
    if tuple(args.development_weeks) != DEVELOPMENT_WEEKS:
        raise ValueError("Development weeks are not exactly Weeks 01-12")
    if tuple(args.frozen_weeks) != FROZEN_WEEKS:
        raise ValueError("Frozen weeks are not exactly Weeks 16-18")

    selection_path = (PROJECT_ROOT / args.selection).resolve()
    audit_path = (PROJECT_ROOT / args.selection_audit).resolve()
    normalized_root = (PROJECT_ROOT / args.normalized_root).resolve()
    cohort_root = (PROJECT_ROOT / args.cohort_root).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    audit_text = audit_path.read_text(encoding="utf-8")
    authorization = "PASS — AUTHORIZE ONE-TIME FROZEN EVALUATION"
    if authorization not in audit_text:
        raise ValueError("Selection audit does not authorize frozen evaluation")
    specifications = validate_frozen_inventory(selection)
    normalized_versions = _manifest_header(
        normalized_root / "manifest.json",
        ("artifact_format_version", "coordinate_transform_version"),
    )
    cohort_versions = _manifest_header(
        cohort_root / "manifest.json", ("artifact_format_version",)
    )

    runtimes: dict[str, float] = {}
    weekly_reconciliation: list[dict[str, Any]] = []
    development_chunks = []
    for week in DEVELOPMENT_WEEKS:
        week_started = time.perf_counter()
        samples, diagnostic = _build_week(
            normalized_root, cohort_root, week
        )
        development_chunks.append(samples)
        weekly_reconciliation.append(diagnostic)
        runtimes[week] = round(time.perf_counter() - week_started, 3)
        gc.collect()
    development = (
        pd.concat(development_chunks, ignore_index=True)
        .sort_values(list(SAMPLE_KEY), kind="stable")
        .reset_index(drop=True)
    )
    del development_chunks
    _validate_samples(
        development,
        expected_weeks=DEVELOPMENT_WEEKS,
        expected_split="development_train",
    )

    fitted: dict[tuple[str, int, str], tuple[Any, Any]] = {}
    development_summaries = []
    selection_counts = {
        (row["population"], row["horizon"]): row
        for row in selection["sample_counts"]
        if row["split"] == "development_train"
    }
    for specification in specifications:
        key = (
            specification["population"],
            specification["horizon"],
            specification["task"],
        )
        group = _population(development, key[0], key[1])
        expected = selection_counts[(key[0], key[1])]
        summary = _summary(group)
        if (
            summary["sample_count"] != expected["sample_count"]
            or summary["play_count"] != expected["play_count"]
        ):
            raise ValueError(f"Development count mismatch for {key}")
        if key[2] == "regression":
            development_summaries.append(
                {"population": key[0], "horizon": key[1], **summary}
            )
        fitted[key] = (
            fit_predictor(specification, group),
            fit_predictor(specification, group, comparator=True),
        )

    frozen_chunks = []
    for week in FROZEN_WEEKS:
        week_started = time.perf_counter()
        samples, diagnostic = _build_week(
            normalized_root, cohort_root, week
        )
        frozen_chunks.append(samples)
        weekly_reconciliation.append(diagnostic)
        runtimes[week] = round(time.perf_counter() - week_started, 3)
        gc.collect()
    frozen = (
        pd.concat(frozen_chunks, ignore_index=True)
        .sort_values(list(SAMPLE_KEY), kind="stable")
        .reset_index(drop=True)
    )
    del frozen_chunks
    _validate_samples(
        frozen,
        expected_weeks=FROZEN_WEEKS,
        expected_split="frozen_test",
    )
    development_plays = set(
        map(tuple, development[["game_id", "play_id"]].to_numpy())
    )
    frozen_plays = set(map(tuple, frozen[["game_id", "play_id"]].to_numpy()))
    if development_plays & frozen_plays:
        raise ValueError("Development and frozen game/play keys overlap")

    evaluations = []
    frozen_summaries = []
    evidence = []
    for specification in specifications:
        key = (
            specification["population"],
            specification["horizon"],
            specification["task"],
        )
        group = _population(frozen, key[0], key[1])
        summary = _summary(group)
        expected_count, expected_plays = EXPECTED_FROZEN_COUNTS[
            (key[0], key[1])
        ]
        if (
            summary["sample_count"] != expected_count
            or summary["play_count"] != expected_plays
        ):
            raise ValueError(f"Frozen count mismatch for {key}")
        if key[2] == "regression":
            frozen_summaries.append(
                {"population": key[0], "horizon": key[1], **summary}
            )
        selected, comparator = fitted[key]
        evaluation = evaluate_specification(
            specification, selected, comparator, group
        )
        validation_difference = float(
            specification["selection_validation_metric"]
            - specification["comparator_validation_metric"]
        )
        primary = "mae" if key[2] == "regression" else "log_loss"
        frozen_difference = evaluation["metric_differences"][primary]
        evidence.append(
            {
                "population": key[0],
                "horizon": key[1],
                "task": key[2],
                "validation_selected_minus_comparator": validation_difference,
                "frozen_selected_minus_comparator": frozen_difference,
                "frozen_primary_metric_improved": frozen_difference < 0,
                "validation_and_frozen_direction_agree": (
                    (validation_difference < 0) == (frozen_difference < 0)
                ),
                "outside_h15": key[1] != 15,
            }
        )
        evaluations.append(
            {
                "population": key[0],
                "horizon": key[1],
                "task": key[2],
                "selected_candidate": specification["selected_candidate"],
                "selected_hyperparameter": specification[
                    "selected_hyperparameter"
                ],
                "comparator_candidate": specification[
                    "comparator_candidate"
                ],
                "comparator_hyperparameter": specification[
                    "comparator_hyperparameter"
                ],
                "frozen_summary": summary,
                **evaluation,
            }
        )

    mismatch_count = sum(
        row["mismatch_count"] for row in weekly_reconciliation
    )
    confirmed = sum(
        row["validation_and_frozen_direction_agree"] for row in evidence
    )
    total_runtime = round(time.perf_counter() - started, 3)
    result = {
        "result_format_version": "milestone_4_frozen_test_result_v1",
        "evaluation_timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "audit_authorization": authorization,
        "selection_artifact": _relative(selection_path),
        "selection_result_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "output_artifact": _relative(output),
        "development_weeks": list(DEVELOPMENT_WEEKS),
        "frozen_weeks": list(FROZEN_WEEKS),
        "validation_weeks_used_for_fitting": [],
        "validation_rows_used_for_fitting": 0,
        "source_artifact_versions": {
            "normalized_tracking": normalized_versions[
                "artifact_format_version"
            ],
            "coordinate_transform": normalized_versions[
                "coordinate_transform_version"
            ],
            "cohorts": cohort_versions["artifact_format_version"],
        },
        "frozen_specifications": specifications,
        "frozen_selection_count": len(specifications),
        "selection_or_comparator_change_count": 0,
        "development_target_summaries": development_summaries,
        "frozen_target_summaries": frozen_summaries,
        "evaluations": evaluations,
        "decision_evidence": {
            "units": evidence,
            "confirmed_direction_count": int(confirmed),
            "reversed_direction_count": int(len(evidence) - confirmed),
            "confirmed_outside_h15_count": int(
                sum(
                    row["validation_and_frozen_direction_agree"]
                    and row["outside_h15"]
                    for row in evidence
                )
            ),
        },
        "bootstrap_configuration": {
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": 0.95,
            "unit": "play",
        },
        "leakage_diagnostics": {
            "status": "PASS",
            "development_weeks_exact": True,
            "frozen_weeks_exact": True,
            "validation_rows_used_for_fitting": 0,
            "cross_split_game_play_count": 0,
            "duplicate_sample_key_count": 0,
            "feature_allowlist_exact": True,
            "prohibited_feature_count": 0,
            "preprocessing_fit_population": "development_train",
            "deterministic_ordering": True,
            "nonfinite_target_or_prediction_count": 0,
        },
        "reconciliation_diagnostics": {
            "status": "PASS",
            "mismatch_count": int(mismatch_count),
            "weekly": weekly_reconciliation,
            "frozen_expected_count_source": (
                "Milestone 3 full-season totals and audited development/"
                "validation counts"
            ),
        },
        "runtime_by_processed_week_seconds": runtimes,
        "total_runtime_seconds": total_runtime,
        "claim_boundary": [
            (
                "Evaluation concerns future separation dynamics for "
                "competition-designated target-defender pairs."
            ),
            "It does not establish official coverage assignments.",
            "It does not establish quarterback decision quality.",
            "It does not estimate pass completion probability.",
            "It does not establish causal defensive effectiveness.",
            "It does not establish complete passing-window openness.",
            "It does not establish betting value.",
        ],
    }
    if mismatch_count:
        raise ValueError("Reconciliation mismatch")
    atomic_write_json(output, result)
    print(json.dumps(_console(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
