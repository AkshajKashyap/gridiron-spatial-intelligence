"""Development-only coefficient stability and fixed validation ablations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .baseline_features import REGISTERED_NUMERIC_FEATURES, feature_matrix
from .baseline_models import (
    DEVELOPMENT_WEEKS,
    VALIDATION_WEEKS,
    build_classification_pipeline,
    build_regression_pipeline,
    classification_metrics,
    regression_metrics,
)


BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 200
PLAY_KEY = ("game_id", "play_id")
ABLATION_ORDER = (
    "full",
    "without_field_location",
    "without_relative_vector",
    "without_defender_context",
    "separation_only",
)
ABLATION_REMOVALS = {
    "without_field_location": (
        "target_x_origin",
        "target_y_origin",
        "defender_x_origin",
        "defender_y_origin",
    ),
    "without_relative_vector": ("dx", "dy", "abs_dx", "abs_dy"),
    "without_defender_context": (
        "valid_observed_defender_count_origin",
        "defender_rank_origin",
        "nearest_observed_defender_indicator",
    ),
}


def validate_interpretation_weeks(
    development_weeks: Sequence[str],
    validation_weeks: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require the registered pre-frozen chronological split."""

    development = tuple(development_weeks)
    validation = tuple(validation_weeks)
    supplied = development + validation
    if any(
        not isinstance(week, str)
        or week not in DEVELOPMENT_WEEKS + VALIDATION_WEEKS
        for week in supplied
    ):
        raise ValueError("Malformed or frozen-test week supplied")
    if development != DEVELOPMENT_WEEKS or validation != VALIDATION_WEEKS:
        raise ValueError(
            "Interpretation requires development Weeks 01-12 and "
            "validation Weeks 13-15 in order"
        )
    return development, validation


def fixed_ablation_feature_sets(
    registered_features: Sequence[str] = REGISTERED_NUMERIC_FEATURES,
) -> dict[str, tuple[str, ...]]:
    """Return the five predeclared feature sets without inventing features."""

    features = tuple(registered_features)
    if features != tuple(REGISTERED_NUMERIC_FEATURES):
        raise ValueError("Registered feature list differs from the frozen list")
    result = {"full": features}
    for name, removals in ABLATION_REMOVALS.items():
        missing = [feature for feature in removals if feature not in features]
        if missing:
            raise ValueError(f"Ablation fields are absent: {missing}")
        result[name] = tuple(
            feature for feature in features if feature not in removals
        )
    result["separation_only"] = ("separation_origin",)
    return {name: result[name] for name in ABLATION_ORDER}


def bootstrap_play_row_positions(
    samples: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample plays with replacement and retain every row in each draw."""

    missing = [column for column in PLAY_KEY if column not in samples]
    if missing:
        raise ValueError(f"Samples are missing play key columns: {missing}")
    if samples.empty:
        raise ValueError("Cannot bootstrap an empty development sample")
    play_index = pd.MultiIndex.from_frame(samples.loc[:, list(PLAY_KEY)])
    codes, unique_plays = pd.factorize(play_index, sort=False)
    draws = rng.integers(0, len(unique_plays), size=len(unique_plays))
    multiplicities = np.bincount(draws, minlength=len(unique_plays))
    return np.repeat(
        np.arange(len(samples), dtype=np.int64), multiplicities[codes]
    )


def summarize_coefficients(
    features: Sequence[str],
    full_development: np.ndarray,
    bootstrap_coefficients: np.ndarray,
) -> list[dict[str, float | str]]:
    """Summarize standardized coefficients in registered feature order."""

    names = tuple(features)
    full = np.asarray(full_development, dtype=float)
    draws = np.asarray(bootstrap_coefficients, dtype=float)
    if full.shape != (len(names),):
        raise ValueError("Full-development coefficient shape is invalid")
    if draws.ndim != 2 or draws.shape[1] != len(names) or not len(draws):
        raise ValueError("Bootstrap coefficient shape is invalid")
    if not np.isfinite(full).all() or not np.isfinite(draws).all():
        raise ValueError("Coefficient output contains non-finite values")
    positive = (draws > 0).mean(axis=0)
    negative = (draws < 0).mean(axis=0)
    return [
        {
            "feature": feature,
            "full_development_standardized_coefficient": float(full[index]),
            "bootstrap_median": float(np.median(draws[:, index])),
            "bootstrap_p10": float(np.quantile(draws[:, index], 0.10)),
            "bootstrap_p90": float(np.quantile(draws[:, index], 0.90)),
            "positive_fraction": float(positive[index]),
            "negative_fraction": float(negative[index]),
            "sign_stability": float(max(positive[index], negative[index])),
        }
        for index, feature in enumerate(names)
    ]


def _pipeline(
    task: str,
    candidate: str,
    hyperparameter: dict[str, float] | None,
):
    if task == "regression":
        if candidate == "multivariable_ols":
            return build_regression_pipeline(candidate)
        if candidate == "ridge_alpha_10":
            if hyperparameter != {"alpha": 10.0}:
                raise ValueError("H15 ridge must retain alpha 10.0")
            return build_regression_pipeline("ridge", alpha=10.0)
    elif task == "classification":
        if not candidate.startswith("multivariable_logistic_c_"):
            raise ValueError("Unsupported selected classification model")
        c_value = None if hyperparameter is None else hyperparameter.get("C")
        return build_classification_pipeline(
            "multivariable_logistic", c_value=c_value
        )
    raise ValueError(f"Unsupported selected specification: {task}/{candidate}")


def _target(samples: pd.DataFrame, task: str) -> pd.Series:
    name = "separation_change" if task == "regression" else "closing"
    if name not in samples:
        raise ValueError(f"Samples are missing exact target: {name}")
    target = pd.to_numeric(samples[name], errors="raise")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target contains non-finite values")
    if task == "classification":
        target = target.astype("int8")
        if set(target.unique()) - {0, 1}:
            raise ValueError("Closing target must be binary")
    return target


def _coefficients(pipeline: Any, feature_count: int) -> np.ndarray:
    coefficients = np.asarray(
        pipeline.named_steps["estimator"].coef_, dtype=float
    ).reshape(-1)
    if coefficients.shape != (feature_count,):
        raise ValueError("Estimator coefficient shape is invalid")
    if not np.isfinite(coefficients).all():
        raise ValueError("Estimator produced non-finite coefficients")
    return coefficients


def coefficient_stability(
    development: pd.DataFrame,
    *,
    task: str,
    candidate: str,
    hyperparameter: dict[str, float] | None,
    features: Sequence[str] = REGISTERED_NUMERIC_FEATURES,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Fit and summarize development-play bootstrap coefficients."""

    original = development.copy(deep=True)
    selected = tuple(features)
    if selected != tuple(REGISTERED_NUMERIC_FEATURES):
        raise ValueError("Coefficient analysis requires the full feature list")
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    x = feature_matrix(original, selected)
    y = _target(original, task)
    if task == "classification" and y.nunique() != 2:
        raise ValueError("Development classification target needs both classes")
    full_pipeline = _pipeline(task, candidate, hyperparameter)
    full_pipeline.fit(x, y)
    full = _coefficients(full_pipeline, len(selected))

    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, len(selected)), dtype=float)
    for draw in range(resamples):
        positions = bootstrap_play_row_positions(original, rng)
        sampled_y = y.iloc[positions]
        if task == "classification" and sampled_y.nunique() != 2:
            raise ValueError("A play bootstrap resample contains one class")
        pipeline = _pipeline(task, candidate, hyperparameter)
        pipeline.fit(x.iloc[positions], sampled_y)
        draws[draw] = _coefficients(pipeline, len(selected))
    return {
        "bootstrap_seed": int(seed),
        "completed_bootstrap_resamples": int(resamples),
        "sampling_unit": "game_id/play_id",
        "preprocessing_fit_population": "development_train",
        "features": summarize_coefficients(selected, full, draws),
    }


def fixed_ablation_evaluation(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    task: str,
    candidate: str,
    hyperparameter: dict[str, float] | None,
    registered_features: Sequence[str] = REGISTERED_NUMERIC_FEATURES,
) -> list[dict[str, Any]]:
    """Fit fixed variants on development and evaluate validation once."""

    dev = development.copy(deep=True)
    val = validation.copy(deep=True)
    feature_sets = fixed_ablation_feature_sets(registered_features)
    y_dev = _target(dev, task)
    y_val = _target(val, task)
    if task == "classification" and y_dev.nunique() != 2:
        raise ValueError("Development classification target needs both classes")
    records: list[dict[str, Any]] = []
    for name, features in feature_sets.items():
        pipeline = _pipeline(task, candidate, hyperparameter)
        pipeline.fit(feature_matrix(dev, features), y_dev)
        if task == "regression":
            measured = regression_metrics(
                y_val, pipeline.predict(feature_matrix(val, features))
            )
            metrics = {
                key: float(measured[key])
                for key in ("mae", "rmse", "median_absolute_error")
            }
        else:
            measured = classification_metrics(
                y_val,
                pipeline.predict_proba(feature_matrix(val, features))[:, 1],
            )
            metrics = {
                "log_loss": float(measured["log_loss"]),
                "brier_score": float(measured["brier_score"]),
                "roc_auc": (
                    None
                    if measured["roc_auc"] is None
                    else float(measured["roc_auc"])
                ),
            }
        finite = [
            value for value in metrics.values() if value is not None
        ]
        if not np.isfinite(np.asarray(finite, dtype=float)).all():
            raise ValueError("Ablation metrics contain non-finite values")
        records.append(
            {"ablation": name, "features": list(features), **metrics}
        )

    full = records[0]
    for record in records:
        if task == "regression":
            record["mae_difference_vs_full"] = float(
                record["mae"] - full["mae"]
            )
        else:
            record["log_loss_difference_vs_full"] = float(
                record["log_loss"] - full["log_loss"]
            )
            record["brier_difference_vs_full"] = float(
                record["brier_score"] - full["brier_score"]
            )
    return records
