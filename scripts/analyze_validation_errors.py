"""Run fixed origin-separation-bucket validation error analysis."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error


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
    build_classification_pipeline,
    build_regression_pipeline,
)
from gridiron_spatial.error_analysis import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    BUCKETS,
    adequate_bucket_bootstrap,
    assign_separation_buckets,
    classification_bucket_diagnostics,
    empty_bucket_record,
    regression_bucket_diagnostics,
    support_flag,
    validate_binding_comparator,
    validate_error_analysis_weeks,
)
from gridiron_spatial.receiver_defender_pairs import (  # noqa: E402
    build_receiver_defender_pairs,
)

from select_baseline_models import (  # noqa: E402
    COHORT_TABLES,
    TRACKING_COLUMNS,
    _atomic_json,
    _read_cohort_week,
    _relative,
    _split_diagnostics,
)


RESULT_FORMAT_VERSION = "milestone_5_validation_error_v1"
EXPECTED_SELECTED = {
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
EXPECTED_COMPARATORS = {
    ("regression", 5): "single_feature_linear",
    ("regression", 10): "single_feature_linear",
    ("regression", 15): "training_median_constant",
    ("classification", 5): "single_feature_logistic",
    ("classification", 10): "single_feature_logistic",
    ("classification", 15): "single_feature_logistic",
}
EXPECTED_VALIDATION_COUNTS = {
    5: (5211, 2085),
    10: (3499, 1245),
    15: (1620, 502),
}


def _project_path(path: Path) -> Path:
    resolved = (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Path must be inside project: {path}") from error
    return resolved


def _binding_specifications(
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    registered = tuple(selection.get("registered_feature_list", ()))
    if registered != REGISTERED_NUMERIC_FEATURES:
        raise ValueError("Selection feature list differs from registration")
    rows = [
        row
        for row in selection["selections"]
        if row["population"] == "all_pairs"
    ]
    indexed = {(row["task"], int(row["horizon"])): row for row in rows}
    if set(indexed) != set(EXPECTED_SELECTED) or len(rows) != 6:
        raise ValueError("Exactly six all-pair selections are required")
    result = []
    for task in ("regression", "classification"):
        for horizon in HORIZONS:
            row = indexed[(task, horizon)]
            candidate, hyperparameter = EXPECTED_SELECTED[(task, horizon)]
            comparator = row["strongest_constant_or_single_feature"][
                "candidate"
            ]
            if (
                row["selected_candidate"] != candidate
                or row["selected_hyperparameter"] != hyperparameter
                or tuple(row["selected_feature_subset"]) != registered
                or comparator != EXPECTED_COMPARATORS[(task, horizon)]
            ):
                raise ValueError(f"Binding model mismatch for {task}/H{horizon}")
            result.append(
                {
                    "population": "all_pairs",
                    "task": task,
                    "horizon": horizon,
                    "selected_candidate": candidate,
                    "selected_hyperparameter": hyperparameter,
                    "selected_feature_subset": list(registered),
                    "comparator_candidate": validate_binding_comparator(
                        task, comparator
                    ),
                    "comparator_feature_subset": (
                        ["separation_origin"]
                        if "single_feature" in comparator
                        else []
                    ),
                    "preprocessing": row["preprocessing"],
                    "target": (
                        "separation_change"
                        if task == "regression"
                        else "closing"
                    ),
                }
            )
    return result


def _selected_pipeline(specification: dict[str, Any]):
    task = specification["task"]
    candidate = specification["selected_candidate"]
    hyperparameter = specification["selected_hyperparameter"]
    if task == "regression":
        if candidate == "multivariable_ols":
            return build_regression_pipeline(candidate)
        return build_regression_pipeline(
            "ridge", alpha=hyperparameter["alpha"]
        )
    return build_classification_pipeline(
        "multivariable_logistic", c_value=hyperparameter["C"]
    )


def _predictions(
    specification: dict[str, Any],
    development: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    task = specification["task"]
    target_name = specification["target"]
    y_dev = development[target_name]
    features = specification["selected_feature_subset"]
    selected_pipeline = _selected_pipeline(specification)
    selected_pipeline.fit(feature_matrix(development, features), y_dev)
    if task == "regression":
        selected = selected_pipeline.predict(
            feature_matrix(validation, features)
        )
    else:
        selected = selected_pipeline.predict_proba(
            feature_matrix(validation, features)
        )[:, 1]

    comparator_name = specification["comparator_candidate"]
    if comparator_name == "training_mean_constant":
        comparator = np.full(len(validation), float(y_dev.mean()))
    elif comparator_name == "training_median_constant":
        comparator = np.full(len(validation), float(y_dev.median()))
    elif comparator_name == "single_feature_linear":
        pipeline = build_regression_pipeline("single_feature_linear")
        pipeline.fit(
            feature_matrix(development, ("separation_origin",)), y_dev
        )
        comparator = pipeline.predict(
            feature_matrix(validation, ("separation_origin",))
        )
    elif comparator_name == "single_feature_logistic":
        pipeline = build_classification_pipeline(
            "single_feature_logistic"
        )
        pipeline.fit(
            feature_matrix(development, ("separation_origin",)), y_dev
        )
        comparator = pipeline.predict_proba(
            feature_matrix(validation, ("separation_origin",))
        )[:, 1]
    else:
        raise ValueError("Unsupported binding comparator")
    selected = np.asarray(selected, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    if (
        not np.isfinite(selected).all()
        or not np.isfinite(comparator).all()
        or (
            task == "classification"
            and (
                ((selected < 0) | (selected > 1)).any()
                or ((comparator < 0) | (comparator > 1)).any()
            )
        )
    ):
        raise ValueError("Model output is malformed")
    return selected, comparator


def _bucket_results(
    validation: pd.DataFrame,
    selected: np.ndarray,
    comparator: np.ndarray,
    *,
    task: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    target_name = "separation_change" if task == "regression" else "closing"
    target = validation[target_name].to_numpy()
    if task == "regression":
        overall_difference = float(
            mean_absolute_error(target, selected)
            - mean_absolute_error(target, comparator)
        )
        overall = {"mae_difference": overall_difference}
    else:
        labels = target.astype(np.int8)
        overall_difference = float(
            log_loss(labels, selected, labels=[0, 1])
            - log_loss(labels, comparator, labels=[0, 1])
        )
        overall = {
            "log_loss_difference": overall_difference,
            "brier_difference": float(
                brier_score_loss(labels, selected)
                - brier_score_loss(labels, comparator)
            ),
        }
    assigned = assign_separation_buckets(validation["separation_origin"])
    records = []
    for bucket in BUCKETS:
        mask = assigned.eq(bucket["bucket"]).to_numpy()
        if not mask.any():
            records.append(empty_bucket_record(bucket))
            continue
        keys = validation.loc[mask, ["game_id", "play_id"]].reset_index(
            drop=True
        )
        bucket_target = target[mask]
        bucket_selected = selected[mask]
        bucket_comparator = comparator[mask]
        plays = int(keys.drop_duplicates().shape[0])
        support = support_flag(int(mask.sum()), plays)
        if task == "regression":
            diagnostics = regression_bucket_diagnostics(
                keys,
                bucket_target,
                bucket_selected,
                bucket_comparator,
                overall_mae_difference=overall_difference,
            )
        else:
            diagnostics = classification_bucket_diagnostics(
                keys,
                bucket_target,
                bucket_selected,
                bucket_comparator,
                overall_log_loss_difference=overall_difference,
            )
        bootstrap = adequate_bucket_bootstrap(
            keys,
            bucket_target,
            bucket_selected,
            bucket_comparator,
            task=task,
            support=support,
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        records.append(
            {
                **bucket,
                "sample_count": int(mask.sum()),
                "play_count": plays,
                "support_flag": support,
                "observed_mean_separation_change": float(
                    validation.loc[mask, "separation_change"].mean()
                ),
                "diagnostics": diagnostics,
                "bootstrap": bootstrap,
            }
        )
    if sum(row["sample_count"] for row in records) != len(validation):
        raise ValueError("Bucket sample counts do not reconcile")
    return records, overall


def _console(result: dict[str, Any]) -> dict[str, Any]:
    regression = {
        (row["horizon"], bucket["bucket"]): bucket
        for row in result["regression_diagnostics"]
        for bucket in row["buckets"]
    }
    classification = {
        (row["horizon"], bucket["bucket"]): bucket
        for row in result["classification_diagnostics"]
        for bucket in row["buckets"]
    }
    rows = []
    losses = []
    for horizon in HORIZONS:
        for name in (bucket["bucket"] for bucket in BUCKETS):
            reg = regression[(horizon, name)]
            cls = classification[(horizon, name)]
            reg_diag = reg["diagnostics"]
            cls_diag = cls["diagnostics"]
            record = {
                "horizon": horizon,
                "bucket": name,
                "support_flag": reg["support_flag"],
                "samples": reg["sample_count"],
                "plays": reg["play_count"],
                "observed_mean_separation_change": reg[
                    "observed_mean_separation_change"
                ],
                "selected_regression_mae": (
                    None
                    if reg_diag is None
                    else reg_diag["selected_model"]["mae"]
                ),
                "comparator_regression_mae": (
                    None
                    if reg_diag is None
                    else reg_diag["comparator"]["mae"]
                ),
                "regression_mae_difference": (
                    None
                    if reg_diag is None
                    else reg_diag[
                        "mae_difference_selected_minus_comparator"
                    ]
                ),
                "selected_classification_log_loss": (
                    None
                    if cls_diag is None
                    else cls_diag["selected_model"]["log_loss"]
                ),
                "comparator_classification_log_loss": (
                    None
                    if cls_diag is None
                    else cls_diag["comparator"]["log_loss"]
                ),
                "classification_log_loss_difference": (
                    None
                    if cls_diag is None
                    else cls_diag[
                        "log_loss_difference_selected_minus_comparator"
                    ]
                ),
                "selected_probability_bias": (
                    None
                    if cls_diag is None
                    else cls_diag["selected_model"]["probability_bias"]
                ),
                "false_positive_rate": (
                    None
                    if cls_diag is None
                    else cls_diag["selected_confusion_at_0_5"][
                        "false_positive_rate"
                    ]
                ),
                "false_negative_rate": (
                    None
                    if cls_diag is None
                    else cls_diag["selected_confusion_at_0_5"][
                        "false_negative_rate"
                    ]
                ),
                "regression_bootstrap": reg["bootstrap"],
                "classification_bootstrap": cls["bootstrap"],
            }
            rows.append(record)
            if reg_diag and not reg_diag["selected_beats_comparator"]:
                losses.append(
                    {"task": "regression", "horizon": horizon, "bucket": name}
                )
            if cls_diag and not cls_diag["selected_beats_comparator"]:
                losses.append(
                    {
                        "task": "classification",
                        "horizon": horizon,
                        "bucket": name,
                    }
                )
    return {
        "processed_weeks": result["processed_weeks"],
        "frozen_weeks_accessed": 0,
        "validation_rows_used_for_fitting": 0,
        "horizon_reconciliation": result["horizon_counts"],
        "bucket_summary": rows,
        "selected_model_losses": losses,
        "leakage_validation": result["leakage_diagnostics"]["status"],
        "reconciliation_mismatch_count": result[
            "reconciliation_diagnostics"
        ]["mismatch_count"],
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
    development_weeks, validation_weeks = validate_error_analysis_weeks(
        args.development_weeks, args.validation_weeks
    )
    weeks = development_weeks + validation_weeks
    selection_path = _project_path(args.selection)
    normalized_root = _project_path(args.normalized_root)
    cohort_root = _project_path(args.cohort_root)
    output = _project_path(args.output)
    selection_bytes = selection_path.read_bytes()
    selection_checksum = hashlib.sha256(selection_bytes).hexdigest()
    selection = json.loads(selection_bytes)
    specifications = _binding_specifications(selection)
    for binding_name in (
        "model_interpretation_summary.json",
        "classifier_calibration_summary.json",
    ):
        binding = json.loads(
            (
                PROJECT_ROOT / "artifacts/milestone_5" / binding_name
            ).read_bytes()
        )
        if (
            tuple(binding["processed_weeks"]) != weeks
            or binding["frozen_test_weeks_accessed"] != 0
            or binding["binding_milestone_4_selection_checksum"]["value"]
            != selection_checksum
        ):
            raise ValueError(f"Binding artifact mismatch: {binding_name}")

    chunks = []
    weekly = []
    runtimes = {}
    for week in weeks:
        week_started = time.perf_counter()
        tracking_path = normalized_root / f"normalized_{week}.parquet"
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
    if samples.duplicated(list(SAMPLE_KEY)).any():
        raise ValueError("Duplicate pair-horizon keys")
    split = _split_diagnostics(samples)
    if tuple(feature_matrix(samples).columns) != REGISTERED_NUMERIC_FEATURES:
        raise ValueError("Feature matrix differs from registration")

    horizon_counts = []
    regression_results = []
    classification_results = []
    specifications_by_key = {
        (row["task"], row["horizon"]): row for row in specifications
    }
    for horizon in HORIZONS:
        horizon_rows = samples.loc[samples["horizon"].eq(horizon)]
        development = horizon_rows.loc[
            horizon_rows["split"].eq("development_train")
        ]
        validation = horizon_rows.loc[
            horizon_rows["split"].eq("validation")
        ].reset_index(drop=True)
        count = (
            int(len(validation)),
            int(
                validation[["game_id", "play_id"]]
                .drop_duplicates()
                .shape[0]
            ),
        )
        if count != EXPECTED_VALIDATION_COUNTS[horizon]:
            raise ValueError(f"Validation count mismatch at H{horizon}")
        horizon_counts.append(
            {
                "horizon": horizon,
                "sample_count": count[0],
                "play_count": count[1],
            }
        )
        for task, destination in (
            ("regression", regression_results),
            ("classification", classification_results),
        ):
            specification = specifications_by_key[(task, horizon)]
            selected, comparator = _predictions(
                specification, development, validation
            )
            buckets, overall = _bucket_results(
                validation, selected, comparator, task=task
            )
            destination.append(
                {
                    "horizon": horizon,
                    "specification": specification,
                    "overall_validation_difference": overall,
                    "buckets": buckets,
                }
            )

    result = {
        "result_format_version": RESULT_FORMAT_VERSION,
        "generation_timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "output_artifact": _relative(output),
        "processed_weeks": list(weeks),
        "development_weeks": list(development_weeks),
        "validation_weeks": list(validation_weeks),
        "frozen_test_weeks_accessed": 0,
        "binding_milestone_4_selection_checksum": {
            "algorithm": "sha256",
            "value": selection_checksum,
        },
        "binding_specifications_and_comparators": specifications,
        "fixed_bucket_definitions": list(BUCKETS),
        "fixed_support_flag_definitions": {
            "adequate": "sample_count >= 500 and play_count >= 150",
            "limited": (
                "sample_count >= 100 and play_count >= 40, below adequate"
            ),
            "sparse": "otherwise",
        },
        "horizon_counts": horizon_counts,
        "regression_diagnostics": regression_results,
        "classification_diagnostics": classification_results,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples_per_adequate_bucket": BOOTSTRAP_RESAMPLES,
        "leakage_diagnostics": {
            "status": "PASS",
            "processed_weeks_exact": True,
            "frozen_test_weeks_accessed": 0,
            "analyzed_model_count": 6,
            "all_pair_population_only": True,
            "exact_registered_features": True,
            "exact_registered_comparators": True,
            "prohibited_model_matrix_field_count": 0,
            "development_only_preprocessing_fit": True,
            "validation_rows_used_for_fit": 0,
            "input_mutation_detected": False,
            **split,
        },
        "reconciliation_diagnostics": {
            "status": "PASS",
            "mismatch_count": 0,
            "duplicate_pair_horizon_key_count": 0,
            "weekly": weekly,
        },
        "runtime_by_week_seconds": runtimes,
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "claim_boundary": [
            (
                "This analysis describes validation error heterogeneity by "
                "initial target-defender separation."
            ),
            "It does not establish causal effects of separation.",
            "It does not establish official coverage responsibility.",
            "It does not establish quarterback decision quality.",
            "It does not estimate pass completion probability.",
            "It does not establish complete passing-window openness.",
            "It does not establish future-season calibration.",
            "It does not establish betting or deployment value.",
            "It provides no new frozen-test evidence.",
        ],
    }
    _atomic_json(output, result)
    print(json.dumps(_console(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
