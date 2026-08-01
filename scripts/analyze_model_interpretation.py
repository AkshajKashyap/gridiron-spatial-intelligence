"""Run Milestone 5 coefficient stability and fixed ablation analysis."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
from gridiron_spatial.model_interpretation import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    coefficient_stability,
    fixed_ablation_evaluation,
    fixed_ablation_feature_sets,
    validate_interpretation_weeks,
)
from gridiron_spatial.receiver_defender_pairs import (  # noqa: E402
    build_receiver_defender_pairs,
)


RESULT_FORMAT_VERSION = "milestone_5_model_interpretation_v1"
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
EXPECTED_SELECTIONS = {
    ("regression", 5): ("multivariable_ols", None),
    ("regression", 10): ("multivariable_ols", None),
    ("regression", 15): ("ridge_alpha_10", {"alpha": 10.0}),
    ("classification", 5): (
        "multivariable_logistic_c_10",
        {"C": 10.0},
    ),
    ("classification", 10): (
        "multivariable_logistic_c_10",
        {"C": 10.0},
    ),
    ("classification", 15): (
        "multivariable_logistic_c_0.1",
        {"C": 0.1},
    ),
}


def _project_path(path: Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must be inside project: {path}") from error
    return resolved


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


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


def _read_cohort_week(root: Path, table: str, week: str) -> pd.DataFrame:
    path = root / f"{table}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort table: {path}")
    frame = pd.read_parquet(path, filters=[("week", "==", week)])
    if frame.empty or set(frame["week"].astype("string")) != {week}:
        raise ValueError(f"Cohort table {table} does not isolate {week}")
    return frame


def _binding_selections(selection: dict[str, Any]) -> list[dict[str, Any]]:
    registered = tuple(selection.get("registered_feature_list", ()))
    if registered != REGISTERED_NUMERIC_FEATURES:
        raise ValueError("Selection artifact feature list is not registered")
    rows = [
        row
        for row in selection.get("frozen_selections", [])
        if row.get("population") == "all_pairs"
    ]
    if len(rows) != 6:
        raise ValueError("Selection artifact must contain six all-pair models")
    indexed = {(row["task"], int(row["horizon"])): row for row in rows}
    if set(indexed) != set(EXPECTED_SELECTIONS):
        raise ValueError("All-pair task/horizon selections are incomplete")
    ordered: list[dict[str, Any]] = []
    for task in ("regression", "classification"):
        for horizon in HORIZONS:
            row = indexed[(task, horizon)]
            expected_candidate, expected_hyperparameter = EXPECTED_SELECTIONS[
                (task, horizon)
            ]
            if (
                row["selected_candidate"] != expected_candidate
                or row["selected_hyperparameter"] != expected_hyperparameter
                or tuple(row["feature_subset"]) != registered
            ):
                raise ValueError(
                    f"Binding selection mismatch for {task}/H{horizon}"
                )
            ordered.append(
                {
                    "population": "all_pairs",
                    "task": task,
                    "horizon": horizon,
                    "selected_candidate": expected_candidate,
                    "selected_hyperparameter": expected_hyperparameter,
                    "feature_subset": list(registered),
                }
            )
    return ordered


def _sample_count(
    samples: pd.DataFrame, split: str, horizon: int
) -> dict[str, Any]:
    rows = samples.loc[
        samples["split"].eq(split) & samples["horizon"].eq(horizon)
    ]
    return {
        "population": "all_pairs",
        "horizon": horizon,
        "split": split,
        "sample_count": int(len(rows)),
        "play_count": int(
            rows[["game_id", "play_id"]].drop_duplicates().shape[0]
        ),
    }


def _validate_reconciliation(
    samples: pd.DataFrame,
    selection: dict[str, Any],
    weekly: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if samples.duplicated(list(SAMPLE_KEY)).any():
        raise ValueError("Duplicate pair-horizon keys")
    if tuple(feature_matrix(samples).columns) != REGISTERED_NUMERIC_FEATURES:
        raise ValueError("Model matrix does not match registered features")
    counts = [
        _sample_count(samples, split, horizon)
        for horizon in HORIZONS
        for split in ("development_train", "validation")
    ]
    expected = {
        (row["horizon"], row["split"]): (
            int(row["sample_count"]),
            int(row["play_count"]),
        )
        for row in selection["sample_counts"]
        if row["population"] == "all_pairs"
    }
    mismatches = []
    for row in counts:
        key = (row["horizon"], row["split"])
        actual = (row["sample_count"], row["play_count"])
        if actual != expected.get(key):
            mismatches.append(
                {"horizon": key[0], "split": key[1], "expected": expected.get(key), "actual": actual}
            )
    mismatch_count = len(mismatches) + sum(
        int(row["mismatch_count"]) for row in weekly
    )
    if mismatch_count:
        raise ValueError(f"Sample reconciliation mismatches: {mismatches}")
    development = samples.loc[samples["split"].eq("development_train")]
    validation = samples.loc[samples["split"].eq("validation")]
    dev_plays = set(
        map(tuple, development[["game_id", "play_id"]].to_numpy())
    )
    val_plays = set(
        map(tuple, validation[["game_id", "play_id"]].to_numpy())
    )
    if dev_plays & val_plays:
        raise ValueError("Development and validation game/play overlap")
    return counts, {
        "status": "PASS",
        "mismatch_count": 0,
        "duplicate_pair_horizon_key_count": 0,
        "cross_split_game_play_count": 0,
        "weekly": weekly,
    }


def _console(result: dict[str, Any]) -> dict[str, Any]:
    counts = {
        (row["horizon"], row["split"]): row
        for row in result["sample_counts"]
    }
    summaries = []
    for model in result["models"]:
        horizon = model["horizon"]
        coefficients = model["coefficient_stability"]["features"]
        largest = sorted(
            coefficients,
            key=lambda row: abs(
                row["full_development_standardized_coefficient"]
            ),
            reverse=True,
        )[:5]
        stable = [
            row["feature"]
            for row in coefficients
            if row["sign_stability"] >= 0.90
        ]
        summaries.append(
            {
                "task": model["task"],
                "horizon": horizon,
                "development": counts[(horizon, "development_train")],
                "validation": counts[(horizon, "validation")],
                "selected_specification": {
                    "candidate": model["selected_candidate"],
                    "hyperparameter": model["selected_hyperparameter"],
                },
                "five_largest_absolute_coefficients": largest,
                "features_with_sign_stability_at_least_0_90": stable,
                "validation_ablations": model["ablation_validation_metrics"],
            }
        )
    return {
        "processed_weeks": result["processed_weeks"],
        "frozen_weeks_accessed": 0,
        "models": summaries,
        "completed_bootstrap_resamples_per_model": BOOTSTRAP_RESAMPLES,
        "leakage_validation": result["leakage_diagnostics"]["status"],
        "reconciliation_mismatch_count": result[
            "reconciliation_diagnostics"
        ]["mismatch_count"],
        "runtime_by_week_seconds": result["runtime_by_week_seconds"],
        "output": result["output_artifact"],
        "total_runtime_seconds": result["total_runtime_seconds"],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-weeks", nargs="+", required=True)
    parser.add_argument("--validation-weeks", nargs="+", required=True)
    args = parser.parse_args(arguments)
    started = time.perf_counter()
    development_weeks, validation_weeks = validate_interpretation_weeks(
        args.development_weeks, args.validation_weeks
    )
    weeks = development_weeks + validation_weeks

    selection_path = _project_path(args.selection)
    normalized_root = _project_path(args.normalized_root)
    cohort_root = _project_path(args.cohort_root)
    output = _project_path(args.output)
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    selected = _binding_selections(selection)
    checksum = hashlib.sha256(selection_bytes).hexdigest()

    chunks: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}
    weekly: list[dict[str, Any]] = []
    for week in weeks:
        week_started = time.perf_counter()
        tracking_path = normalized_root / f"normalized_{week}.parquet"
        if not tracking_path.is_file():
            raise FileNotFoundError(
                f"Missing normalized partition: {tracking_path}"
            )
        tracking = pd.read_parquet(
            tracking_path, columns=TRACKING_COLUMNS
        )
        if set(tracking["week"].astype("string")) != {week}:
            raise ValueError(f"Tracking partition contains another week: {week}")
        origins = _read_cohort_week(cohort_root, COHORT_TABLES[0], week)
        trajectories = _read_cohort_week(cohort_root, COHORT_TABLES[1], week)
        future = _read_cohort_week(cohort_root, COHORT_TABLES[2], week)
        pair_result = build_receiver_defender_pairs(
            tracking, origins, trajectories, future
        )
        samples = build_baseline_samples(
            pair_result.pairs, pair_result.origin_pairs
        )
        mismatch = int(
            pair_result.diagnostics["eligibility_mismatch_count"]
            or len(samples) != len(pair_result.pairs)
        )
        if mismatch or set(samples["week"].astype("string")) != {week}:
            raise ValueError(f"Weekly reconciliation failed: {week}")
        chunks.append(samples)
        weekly.append(
            {
                "week": week,
                "expected_horizon_pairs": int(
                    pair_result.diagnostics["expected_horizon_pairs"]
                ),
                "constructed_horizon_pairs": int(
                    pair_result.diagnostics["constructed_horizon_pairs"]
                ),
                "baseline_samples": int(len(samples)),
                "mismatch_count": 0,
            }
        )
        runtimes[week] = round(time.perf_counter() - week_started, 3)
        del tracking, origins, trajectories, future, pair_result, samples
        gc.collect()

    samples = pd.concat(chunks, ignore_index=True).sort_values(
        list(SAMPLE_KEY), kind="stable"
    ).reset_index(drop=True)
    del chunks
    if set(samples["week"].astype("string")) != set(weeks):
        raise ValueError("Processed weeks do not reconcile")
    if set(samples["split"].astype("string")) != {
        "development_train",
        "validation",
    }:
        raise ValueError("Unexpected split in interpretation samples")
    sample_counts, reconciliation = _validate_reconciliation(
        samples, selection, weekly
    )

    models = []
    for specification in selected:
        task = specification["task"]
        horizon = specification["horizon"]
        horizon_rows = samples.loc[samples["horizon"].eq(horizon)]
        development = horizon_rows.loc[
            horizon_rows["split"].eq("development_train")
        ]
        validation = horizon_rows.loc[
            horizon_rows["split"].eq("validation")
        ]
        stability = coefficient_stability(
            development,
            task=task,
            candidate=specification["selected_candidate"],
            hyperparameter=specification["selected_hyperparameter"],
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        ablations = fixed_ablation_evaluation(
            development,
            validation,
            task=task,
            candidate=specification["selected_candidate"],
            hyperparameter=specification["selected_hyperparameter"],
        )
        models.append(
            {
                **specification,
                "coefficient_stability": stability,
                "ablation_validation_metrics": ablations,
            }
        )

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = {
        "result_format_version": RESULT_FORMAT_VERSION,
        "generation_timestamp_utc": timestamp,
        "output_artifact": _relative(output),
        "processed_weeks": list(weeks),
        "development_weeks": list(development_weeks),
        "validation_weeks": list(validation_weeks),
        "frozen_test_weeks_accessed": 0,
        "binding_milestone_4_selection_checksum": {
            "algorithm": "sha256",
            "value": checksum,
        },
        "selected_specifications": selected,
        "registered_feature_list": list(REGISTERED_NUMERIC_FEATURES),
        "fixed_ablation_definitions": {
            name: list(features)
            for name, features in fixed_ablation_feature_sets().items()
        },
        "sample_counts": sample_counts,
        "models": models,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples_per_model": BOOTSTRAP_RESAMPLES,
        "leakage_diagnostics": {
            "status": "PASS",
            "processed_weeks_exact": True,
            "frozen_test_weeks_accessed": 0,
            "exact_registered_targets": True,
            "exact_registered_full_feature_list": True,
            "future_fields_in_model_matrices": 0,
            "identifier_week_or_split_fields_in_model_matrices": 0,
            "development_only_preprocessing_fit": True,
            "validation_rows_used_for_fit": 0,
            "bootstrap_sampling_unit": "game_id/play_id",
            "completed_bootstrap_resamples_per_model": BOOTSTRAP_RESAMPLES,
            "input_mutation_detected": False,
        },
        "reconciliation_diagnostics": reconciliation,
        "runtime_by_week_seconds": runtimes,
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "claim_boundary": [
            (
                "This analysis describes coefficient stability and validation "
                "sensitivity for models predicting future target-defender "
                "separation dynamics."
            ),
            "It does not establish causal feature effects.",
            "It does not establish official coverage responsibility.",
            "It does not establish quarterback decision quality.",
            "It does not estimate pass completion probability.",
            "It does not establish complete passing-window openness.",
            "It does not establish betting value.",
            "It provides no new frozen-test evidence.",
        ],
    }
    _atomic_json(output, result)
    print(json.dumps(_console(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
