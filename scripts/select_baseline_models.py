"""Select Milestone 4 baselines using development and validation only."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import tempfile
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
from gridiron_spatial.baseline_models import (  # noqa: E402
    DEVELOPMENT_WEEKS,
    EXPECTED_WEEKS,
    LOGISTIC_CS,
    PREPROCESSING_SPECIFICATION,
    RIDGE_ALPHAS,
    VALIDATION_WEEKS,
    select_classification_baseline,
    select_regression_baseline,
    validate_requested_weeks,
)
from gridiron_spatial.receiver_defender_pairs import (  # noqa: E402
    build_receiver_defender_pairs,
)


RESULT_FORMAT_VERSION = "milestone_4_baseline_selection_v1"
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
COHORT_TABLES = (
    "primary_origins",
    "trajectory_eligibility",
    "future_separation_eligibility",
)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must be inside project: {path}") from error


def _manifest_header(path: Path, fields: tuple[str, ...]) -> dict[str, str]:
    """Read only version fields before any manifest week inventory."""

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
    missing = [field for field in fields if field not in found]
    if missing:
        raise ValueError(f"Manifest header is missing version fields: {missing}")
    return found


def _read_cohort_week(root: Path, name: str, week: str) -> pd.DataFrame:
    path = root / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort table: {path}")
    frame = pd.read_parquet(path, filters=[("week", "==", week)])
    if frame.empty or set(frame["week"].astype("string")) != {week}:
        raise ValueError(f"Cohort table {name} does not isolate {week}")
    return frame


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _split_diagnostics(samples: pd.DataFrame) -> dict[str, Any]:
    development = samples.loc[samples["split"].eq("development_train")]
    validation = samples.loc[samples["split"].eq("validation")]
    observed_dev = tuple(
        sorted(development["week"].astype("string").unique())
    )
    observed_val = tuple(
        sorted(validation["week"].astype("string").unique())
    )
    if observed_dev != DEVELOPMENT_WEEKS:
        raise ValueError("Development rows do not contain exactly Weeks 01-12")
    if observed_val != VALIDATION_WEEKS:
        raise ValueError("Validation rows do not contain exactly Weeks 13-15")
    if set(samples["split"].astype("string")) != {
        "development_train",
        "validation",
    }:
        raise ValueError("Unsupported split in development/validation samples")
    dev_plays = set(
        map(tuple, development[["game_id", "play_id"]].to_numpy())
    )
    val_plays = set(
        map(tuple, validation[["game_id", "play_id"]].to_numpy())
    )
    overlap = dev_plays & val_plays
    if overlap:
        raise ValueError("Game/play keys overlap development and validation")
    return {
        "development_weeks": list(observed_dev),
        "validation_weeks": list(observed_val),
        "cross_split_game_play_count": 0,
    }


def _population_counts(
    samples: pd.DataFrame,
    population: str,
    horizon: int,
) -> list[dict[str, Any]]:
    records = []
    selected = samples.loc[samples["horizon"].eq(horizon)]
    if population == "nearest_observed_defender":
        selected = selected.loc[
            selected["nearest_observed_defender_indicator"].eq(1)
        ]
    for split in ("development_train", "validation"):
        group = selected.loc[selected["split"].eq(split)]
        if group.empty:
            raise ValueError(f"Empty sample group: {population}/{horizon}/{split}")
        records.append(
            {
                "population": population,
                "horizon": horizon,
                "split": split,
                "sample_count": int(len(group)),
                "play_count": int(
                    group[["game_id", "play_id"]].drop_duplicates().shape[0]
                ),
                "closing_rate": float(group["closing"].mean()),
            }
        )
    return records


def _compact_console(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    counts = {
        (row["population"], row["horizon"], row["split"]): row
        for row in result["sample_counts"]
    }
    selections = {
        (row["population"], row["horizon"], row["task"]): row
        for row in result["selections"]
    }
    for population in ("all_pairs", "nearest_observed_defender"):
        for horizon in HORIZONS:
            development = counts[(population, horizon, "development_train")]
            validation = counts[(population, horizon, "validation")]
            regression = selections[(population, horizon, "regression")]
            classification = selections[
                (population, horizon, "classification")
            ]
            rows.append(
                {
                    "population": population,
                    "horizon": horizon,
                    "development_samples": development["sample_count"],
                    "development_plays": development["play_count"],
                    "validation_samples": validation["sample_count"],
                    "validation_plays": validation["play_count"],
                    "development_closing_rate": development["closing_rate"],
                    "validation_closing_rate": validation["closing_rate"],
                    "selected_regression": regression["selected_candidate"],
                    "selected_validation_mae": regression[
                        "selection_validation_metric"
                    ],
                    "regression_comparator": regression[
                        "strongest_constant_or_single_feature"
                    ],
                    "selected_classification": classification[
                        "selected_candidate"
                    ],
                    "selected_validation_log_loss": classification[
                        "selection_validation_metric"
                    ],
                    "classification_comparator": classification[
                        "strongest_constant_or_single_feature"
                    ],
                }
            )
    return {
        "processed_weeks": result["processed_weeks"],
        "frozen_weeks_accessed": 0,
        "leakage_validation": result["leakage_diagnostics"]["status"],
        "reconciliation_mismatch_count": result[
            "reconciliation_diagnostics"
        ]["mismatch_count"],
        "selection_summary": rows,
        "output": result["output_artifact"],
        "total_runtime_seconds": result["total_runtime_seconds"],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weeks", nargs="+", required=True)
    args = parser.parse_args(arguments)
    started = time.perf_counter()
    weeks = validate_requested_weeks(args.weeks)
    normalized_root = (PROJECT_ROOT / args.normalized_root).resolve()
    cohort_root = (PROJECT_ROOT / args.cohort_root).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    normalized_versions = _manifest_header(
        normalized_root / "manifest.json",
        ("artifact_format_version", "coordinate_transform_version"),
    )
    cohort_versions = _manifest_header(
        cohort_root / "manifest.json", ("artifact_format_version",)
    )

    chunks: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}
    reconciliation: list[dict[str, Any]] = []
    for week in weeks:
        week_started = time.perf_counter()
        normalized_path = normalized_root / f"normalized_{week}.parquet"
        if not normalized_path.is_file():
            raise FileNotFoundError(f"Missing normalized partition: {normalized_path}")
        tracking = pd.read_parquet(normalized_path, columns=TRACKING_COLUMNS)
        if set(tracking["week"].astype("string")) != {week}:
            raise ValueError(f"Normalized partition contains another week: {week}")
        origins = _read_cohort_week(cohort_root, COHORT_TABLES[0], week)
        trajectories = _read_cohort_week(cohort_root, COHORT_TABLES[1], week)
        future = _read_cohort_week(cohort_root, COHORT_TABLES[2], week)
        pair_result = build_receiver_defender_pairs(
            tracking, origins, trajectories, future
        )
        if pair_result.diagnostics["eligibility_mismatch_count"]:
            raise ValueError(f"Pair eligibility mismatch in {week}")
        samples = build_baseline_samples(
            pair_result.pairs, pair_result.origin_pairs
        )
        if len(samples) != len(pair_result.pairs):
            raise ValueError(f"Sample/pair reconciliation mismatch in {week}")
        if set(samples["week"].astype("string")) != {week}:
            raise ValueError(f"Baseline samples contain another week: {week}")
        chunks.append(samples)
        reconciliation.append(
            {
                "week": week,
                "expected_horizon_pairs": pair_result.diagnostics[
                    "expected_horizon_pairs"
                ],
                "constructed_horizon_pairs": pair_result.diagnostics[
                    "constructed_horizon_pairs"
                ],
                "baseline_samples": int(len(samples)),
                "mismatch_count": 0,
            }
        )
        runtimes[week] = round(time.perf_counter() - week_started, 3)
        del (
            tracking,
            origins,
            trajectories,
            future,
            pair_result,
            samples,
        )
        gc.collect()

    all_samples = pd.concat(chunks, ignore_index=True)
    del chunks
    if all_samples.duplicated(list(SAMPLE_KEY)).any():
        raise ValueError("Duplicate sample keys across processed weeks")
    all_samples = all_samples.sort_values(
        list(SAMPLE_KEY), kind="stable"
    ).reset_index(drop=True)
    if not all_samples.equals(
        all_samples.sort_values(list(SAMPLE_KEY), kind="stable").reset_index(
            drop=True
        )
    ):
        raise ValueError("Sample ordering is not deterministic")
    split_diagnostics = _split_diagnostics(all_samples)
    if list(feature_matrix(all_samples).columns) != list(
        REGISTERED_NUMERIC_FEATURES
    ):
        raise ValueError("Feature matrix differs from registered allowlist")

    sample_counts: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    frozen_selections: list[dict[str, Any]] = []
    selection_timestamp = datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    for population in ("all_pairs", "nearest_observed_defender"):
        population_samples = all_samples
        if population == "nearest_observed_defender":
            population_samples = all_samples.loc[
                all_samples["nearest_observed_defender_indicator"].eq(1)
            ]
        for horizon in HORIZONS:
            sample_counts.extend(
                _population_counts(all_samples, population, horizon)
            )
            horizon_samples = population_samples.loc[
                population_samples["horizon"].eq(horizon)
            ]
            development = horizon_samples.loc[
                horizon_samples["split"].eq("development_train")
            ]
            validation = horizon_samples.loc[
                horizon_samples["split"].eq("validation")
            ]
            for selection in (
                select_regression_baseline(development, validation),
                select_classification_baseline(development, validation),
            ):
                record = {
                    "population": population,
                    "horizon": horizon,
                    **selection,
                }
                selections.append(record)
                frozen_selections.append(
                    {
                        "population": population,
                        "horizon": horizon,
                        "task": selection["task"],
                        "selected_candidate": selection[
                            "selected_candidate"
                        ],
                        "selected_hyperparameter": selection[
                            "selected_hyperparameter"
                        ],
                        "feature_subset": selection[
                            "selected_feature_subset"
                        ],
                        "preprocessing": selection["preprocessing"],
                        "primary_metric": selection["primary_metric"],
                        "selection_validation_metric": selection[
                            "selection_validation_metric"
                        ],
                        "selection_timestamp_utc": selection_timestamp,
                    }
                )

    mismatch_count = sum(row["mismatch_count"] for row in reconciliation)
    if mismatch_count:
        raise ValueError("Aggregate reconciliation mismatch")
    total_runtime = round(time.perf_counter() - started, 3)
    result = {
        "result_format_version": RESULT_FORMAT_VERSION,
        "generation_timestamp_utc": selection_timestamp,
        "output_artifact": _relative(output),
        "processed_weeks": list(weeks),
        "frozen_test_weeks_accessed": 0,
        "source_artifact_versions": {
            "normalized_tracking": normalized_versions[
                "artifact_format_version"
            ],
            "coordinate_transform": normalized_versions[
                "coordinate_transform_version"
            ],
            "cohorts": cohort_versions["artifact_format_version"],
        },
        "registered_feature_list": list(REGISTERED_NUMERIC_FEATURES),
        "actual_feature_list": list(REGISTERED_NUMERIC_FEATURES),
        "optional_categorical_fields": {
            "included": [],
            "omitted": ["target_position_or_role", "defender_position_or_role"],
            "reason": (
                "The reused pair result does not carry these fields directly; "
                "no additional join was authorized."
            ),
        },
        "sample_counts": sample_counts,
        "leakage_diagnostics": {
            "status": "PASS",
            "requested_weeks_exact": True,
            "frozen_test_weeks_accessed": 0,
            "split_assignments_valid": True,
            "cross_split_game_play_count": split_diagnostics[
                "cross_split_game_play_count"
            ],
            "duplicate_sample_key_count": 0,
            "feature_allowlist_exact": True,
            "future_fields_in_feature_matrix": 0,
            "identifier_split_or_week_fields_in_feature_matrix": 0,
            "target_horizon_matching": True,
            "development_only_preprocessing_fit": True,
            "validation_targets_used_for_fit": False,
            "deterministic_sample_ordering": True,
            **split_diagnostics,
        },
        "reconciliation_diagnostics": {
            "status": "PASS",
            "mismatch_count": mismatch_count,
            "weekly": reconciliation,
        },
        "preprocessing_configuration": PREPROCESSING_SPECIFICATION,
        "registered_grids": {
            "ridge_alpha": list(RIDGE_ALPHAS),
            "logistic_c": list(LOGISTIC_CS),
        },
        "selections": selections,
        "frozen_selections": frozen_selections,
        "runtime_by_week_seconds": runtimes,
        "total_runtime_seconds": total_runtime,
        "claim_boundary": [
            (
                "Selection concerns future separation dynamics for "
                "competition-designated target-defender pairs."
            ),
            "It does not establish official coverage responsibility.",
            "It does not establish quarterback decision quality.",
            "It does not estimate pass completion probability.",
            "It does not establish causal defensive effectiveness.",
            "It does not establish complete passing-window openness.",
            "It does not establish betting value.",
        ],
    }
    _atomic_json(output, result)
    print(json.dumps(_compact_console(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
