"""Frozen-specification reproduction and play-clustered evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_features import (
    REGISTERED_NUMERIC_FEATURES,
    SAMPLE_KEY,
    feature_matrix,
)
from .baseline_models import (
    PREPROCESSING_SPECIFICATION,
    build_classification_pipeline,
    build_regression_pipeline,
    classification_metrics,
    regression_metrics,
)


BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 500
EXPECTED_SELECTIONS = {
    ("all_pairs", 5, "regression"): ("multivariable_ols", None),
    ("all_pairs", 10, "regression"): ("multivariable_ols", None),
    ("all_pairs", 15, "regression"): ("ridge_alpha_10", {"alpha": 10.0}),
    ("nearest_observed_defender", 5, "regression"): (
        "multivariable_ols",
        None,
    ),
    ("nearest_observed_defender", 10, "regression"): (
        "ridge_alpha_10",
        {"alpha": 10.0},
    ),
    ("nearest_observed_defender", 15, "regression"): (
        "multivariable_ols",
        None,
    ),
    ("all_pairs", 5, "classification"): (
        "multivariable_logistic_c_10",
        {"C": 10.0},
    ),
    ("all_pairs", 10, "classification"): (
        "multivariable_logistic_c_10",
        {"C": 10.0},
    ),
    ("all_pairs", 15, "classification"): (
        "multivariable_logistic_c_0.1",
        {"C": 0.1},
    ),
    ("nearest_observed_defender", 5, "classification"): (
        "multivariable_logistic_c_10",
        {"C": 10.0},
    ),
    ("nearest_observed_defender", 10, "classification"): (
        "multivariable_logistic_c_10",
        {"C": 10.0},
    ),
    ("nearest_observed_defender", 15, "classification"): (
        "multivariable_logistic_c_10",
        {"C": 10.0},
    ),
}


@dataclass
class FittedPredictor:
    task: str
    candidate: str
    feature_subset: tuple[str, ...]
    pipeline: Any | None = None
    constant: float | None = None

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            result = np.full(len(samples), self.constant, dtype=float)
        else:
            if self.pipeline is None:
                raise ValueError("Predictor has neither pipeline nor constant")
            features = feature_matrix(samples, self.feature_subset)
            if self.task == "regression":
                values = self.pipeline.predict(features)
            else:
                values = self.pipeline.predict_proba(features)[:, 1]
            result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("Model produced non-finite predictions")
        return result


def validate_frozen_inventory(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Return evaluator-ready specs after strict frozen-record validation."""

    if selection.get("frozen_test_weeks_accessed") != 0:
        raise ValueError("Selection artifact reports frozen-test access")
    frozen = selection.get("frozen_selections")
    selections = selection.get("selections")
    if not isinstance(frozen, list) or not isinstance(selections, list):
        raise ValueError("Selection artifact lacks selection inventories")
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in frozen:
        key = (row["population"], row["horizon"], row["task"])
        if key in indexed:
            raise ValueError(f"Duplicate frozen selection: {key}")
        indexed[key] = row
    if set(indexed) != set(EXPECTED_SELECTIONS):
        raise ValueError("Frozen selection inventory is missing or additional")

    selection_index = {
        (row["population"], row["horizon"], row["task"]): row
        for row in selections
    }
    if set(selection_index) != set(EXPECTED_SELECTIONS):
        raise ValueError("Detailed selection inventory does not reconcile")
    result = []
    for key in sorted(EXPECTED_SELECTIONS):
        frozen_row = indexed[key]
        expected_candidate, expected_hyperparameter = EXPECTED_SELECTIONS[key]
        if frozen_row["selected_candidate"] != expected_candidate:
            raise ValueError(f"Unauthorized candidate for {key}")
        if frozen_row["selected_hyperparameter"] != expected_hyperparameter:
            raise ValueError(f"Unauthorized hyperparameter for {key}")
        if tuple(frozen_row["feature_subset"]) != REGISTERED_NUMERIC_FEATURES:
            raise ValueError(f"Feature-list mismatch for {key}")
        if frozen_row["preprocessing"] != PREPROCESSING_SPECIFICATION:
            raise ValueError(f"Preprocessing mismatch for {key}")
        detailed = selection_index[key]
        if detailed["selected_candidate"] != expected_candidate:
            raise ValueError(f"Detailed candidate mismatch for {key}")
        comparator_name = detailed[
            "strongest_constant_or_single_feature"
        ]["candidate"]
        candidate_index = {
            candidate["candidate"]: candidate
            for candidate in detailed["candidates"]
        }
        if comparator_name not in candidate_index:
            raise ValueError(f"Frozen comparator missing for {key}")
        comparator = candidate_index[comparator_name]
        if key[2] == "regression":
            allowed = {
                "training_mean_constant",
                "training_median_constant",
                "single_feature_linear",
            }
            comparator_metric = detailed[
                "strongest_constant_or_single_feature"
            ]["validation_mae"]
        else:
            allowed = {
                "training_closing_rate_constant",
                "single_feature_logistic",
            }
            comparator_metric = detailed[
                "strongest_constant_or_single_feature"
            ]["validation_log_loss"]
        if comparator_name not in allowed:
            raise ValueError(f"Unauthorized frozen comparator for {key}")
        result.append(
            {
                **frozen_row,
                "selected_feature_subset": frozen_row["feature_subset"],
                "comparator_candidate": comparator_name,
                "comparator_hyperparameter": comparator["hyperparameter"],
                "comparator_feature_subset": comparator["feature_subset"],
                "comparator_validation_metric": comparator_metric,
            }
        )
    return result


def fit_predictor(
    specification: dict[str, Any],
    development: pd.DataFrame,
    *,
    comparator: bool = False,
) -> FittedPredictor:
    """Fit one selected or comparator predictor on development only."""

    prefix = "comparator_" if comparator else "selected_"
    candidate = specification[f"{prefix}candidate"]
    hyperparameter = specification[f"{prefix}hyperparameter"]
    features = tuple(specification[f"{prefix}feature_subset"])
    task = specification["task"]
    target_name = "separation_change" if task == "regression" else "closing"
    target = pd.to_numeric(development[target_name], errors="raise")

    if candidate == "training_mean_constant":
        return FittedPredictor(task, candidate, (), constant=float(target.mean()))
    if candidate == "training_median_constant":
        return FittedPredictor(
            task, candidate, (), constant=float(target.median())
        )
    if candidate == "training_closing_rate_constant":
        return FittedPredictor(task, candidate, (), constant=float(target.mean()))
    if candidate == "single_feature_linear":
        pipeline = build_regression_pipeline("single_feature_linear")
    elif candidate == "multivariable_ols":
        pipeline = build_regression_pipeline("multivariable_ols")
    elif candidate.startswith("ridge_alpha_"):
        pipeline = build_regression_pipeline(
            "ridge", alpha=hyperparameter["alpha"]
        )
    elif candidate == "single_feature_logistic":
        pipeline = build_classification_pipeline(
            "single_feature_logistic"
        )
    elif candidate.startswith("multivariable_logistic_c_"):
        pipeline = build_classification_pipeline(
            "multivariable_logistic", c_value=hyperparameter["C"]
        )
    else:
        raise ValueError(f"Unauthorized candidate: {candidate}")
    pipeline.fit(feature_matrix(development, features), target)
    return FittedPredictor(task, candidate, features, pipeline=pipeline)


def play_cluster_interval(
    row_differences: np.ndarray,
    game_ids: pd.Series | np.ndarray,
    play_ids: pd.Series | np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float]:
    """Bootstrap plays while retaining every row from each sampled play."""

    values = np.asarray(row_differences, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Metric differences contain non-finite values")
    grouped = (
        pd.DataFrame(
            {
                "game_id": np.asarray(game_ids),
                "play_id": np.asarray(play_ids),
                "difference": values,
            }
        )
        .groupby(["game_id", "play_id"], sort=True)["difference"]
        .agg(["sum", "count"])
    )
    if grouped.empty:
        raise ValueError("Cannot bootstrap an empty play population")
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(resamples, len(grouped)))
    estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    bounds = np.quantile(estimates, [0.025, 0.975])
    return [float(bounds[0]), float(bounds[1])]


def evaluate_specification(
    specification: dict[str, Any],
    selected: FittedPredictor,
    comparator: FittedPredictor,
    frozen: pd.DataFrame,
) -> dict[str, Any]:
    selected_prediction = selected.predict(frozen)
    comparator_prediction = comparator.predict(frozen)
    task = specification["task"]
    if task == "regression":
        actual = frozen["separation_change"].to_numpy(dtype=float)
        if not np.isfinite(actual).all():
            raise ValueError("Frozen regression target is non-finite")
        selected_metrics = regression_metrics(actual, selected_prediction)
        comparator_metrics = regression_metrics(actual, comparator_prediction)
        row_difference = np.abs(actual - selected_prediction) - np.abs(
            actual - comparator_prediction
        )
        difference = float(
            selected_metrics["mae"] - comparator_metrics["mae"]
        )
        return {
            "selected_metrics": selected_metrics,
            "comparator_metrics": comparator_metrics,
            "metric_differences": {"mae": difference},
            "play_cluster_bootstrap_95_intervals": {
                "mae_difference": play_cluster_interval(
                    row_difference, frozen["game_id"], frozen["play_id"]
                )
            },
        }

    actual = frozen["closing"].to_numpy(dtype=int)
    selected_metrics = classification_metrics(actual, selected_prediction)
    comparator_metrics = classification_metrics(actual, comparator_prediction)
    epsilon = np.finfo(float).eps
    selected_probability = np.clip(
        selected_prediction, epsilon, 1.0 - epsilon
    )
    comparator_probability = np.clip(
        comparator_prediction, epsilon, 1.0 - epsilon
    )
    selected_log = -(
        actual * np.log(selected_probability)
        + (1 - actual) * np.log(1 - selected_probability)
    )
    comparator_log = -(
        actual * np.log(comparator_probability)
        + (1 - actual) * np.log(1 - comparator_probability)
    )
    log_difference_rows = selected_log - comparator_log
    brier_difference_rows = (selected_prediction - actual) ** 2 - (
        comparator_prediction - actual
    ) ** 2
    return {
        "selected_metrics": selected_metrics,
        "comparator_metrics": comparator_metrics,
        "metric_differences": {
            "log_loss": float(
                selected_metrics["log_loss"]
                - comparator_metrics["log_loss"]
            ),
            "brier_score": float(
                selected_metrics["brier_score"]
                - comparator_metrics["brier_score"]
            ),
        },
        "play_cluster_bootstrap_95_intervals": {
            "log_loss_difference": play_cluster_interval(
                log_difference_rows, frozen["game_id"], frozen["play_id"]
            ),
            "brier_score_difference": play_cluster_interval(
                brier_difference_rows, frozen["game_id"], frozen["play_id"]
            ),
        },
    }


def validate_sample_keys(samples: pd.DataFrame) -> None:
    if samples.duplicated(list(SAMPLE_KEY)).any():
        raise ValueError("Duplicate sample keys")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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
