"""Fixed origin-separation-bucket validation error diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    roc_auc_score,
)

from .model_interpretation import (
    bootstrap_play_row_positions,
    validate_interpretation_weeks,
)


BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 300
CONFIDENCE_LEVEL = 0.95
BUCKETS = (
    {"bucket": "0_to_3", "lower": 0.0, "upper": 3.0},
    {"bucket": "3_to_5", "lower": 3.0, "upper": 5.0},
    {"bucket": "5_to_10", "lower": 5.0, "upper": 10.0},
    {"bucket": "10_to_15", "lower": 10.0, "upper": 15.0},
    {"bucket": "15_to_20", "lower": 15.0, "upper": 20.0},
    {"bucket": "20_plus", "lower": 20.0, "upper": None},
)
BUCKET_ORDER = tuple(row["bucket"] for row in BUCKETS)


def validate_error_analysis_weeks(
    development_weeks: Sequence[str],
    validation_weeks: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return validate_interpretation_weeks(
        development_weeks, validation_weeks
    )


def validate_binding_comparator(task: str, candidate: str) -> str:
    allowed = {
        "regression": {
            "training_mean_constant",
            "training_median_constant",
            "single_feature_linear",
        },
        "classification": {
            "training_closing_rate_constant",
            "single_feature_logistic",
        },
    }
    if task not in allowed or candidate not in allowed[task]:
        raise ValueError(f"Unsupported binding comparator: {task}/{candidate}")
    return candidate


def assign_separation_buckets(
    separation: Sequence[float] | pd.Series | np.ndarray,
) -> pd.Series:
    values = pd.to_numeric(pd.Series(separation), errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Origin separation must be finite and nonnegative")
    assigned = pd.cut(
        values,
        bins=[0, 3, 5, 10, 15, 20, np.inf],
        labels=BUCKET_ORDER,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    if assigned.isna().any():
        raise ValueError("Origin separation bucket assignment failed")
    return assigned.astype("string")


def support_flag(sample_count: int, play_count: int) -> str:
    if sample_count >= 500 and play_count >= 150:
        return "adequate"
    if sample_count >= 100 and play_count >= 40:
        return "limited"
    return "sparse"


def _arrays(
    target: Sequence[float] | pd.Series | np.ndarray,
    selected: Sequence[float] | pd.Series | np.ndarray,
    comparator: Sequence[float] | pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=float)
    chosen = np.asarray(selected, dtype=float)
    baseline = np.asarray(comparator, dtype=float)
    if (
        y.ndim != 1
        or chosen.shape != y.shape
        or baseline.shape != y.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(chosen).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("Targets and predictions must be aligned and finite")
    return y, chosen, baseline


def _play_count(keys: pd.DataFrame) -> int:
    if not {"game_id", "play_id"}.issubset(keys):
        raise ValueError("Sample keys require game_id and play_id")
    return int(keys[["game_id", "play_id"]].drop_duplicates().shape[0])


def _regression_metrics(
    keys: pd.DataFrame, target: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "sample_count": int(len(target)),
        "play_count": _play_count(keys),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "median_absolute_error": float(
            median_absolute_error(target, prediction)
        ),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "mean_signed_error": float(error.mean()),
        "median_signed_error": float(np.median(error)),
    }


def regression_bucket_diagnostics(
    keys: pd.DataFrame,
    target: Sequence[float] | pd.Series | np.ndarray,
    selected_prediction: Sequence[float] | pd.Series | np.ndarray,
    comparator_prediction: Sequence[float] | pd.Series | np.ndarray,
    *,
    overall_mae_difference: float,
) -> dict[str, Any]:
    y, selected, comparator = _arrays(
        target, selected_prediction, comparator_prediction
    )
    selected_metrics = _regression_metrics(keys, y, selected)
    comparator_metrics = _regression_metrics(keys, y, comparator)
    difference = float(
        selected_metrics["mae"] - comparator_metrics["mae"]
    )
    absolute = np.abs(selected - y)
    return {
        "selected_model": selected_metrics,
        "comparator": comparator_metrics,
        "mae_difference_selected_minus_comparator": difference,
        "selected_beats_comparator": difference < 0,
        "reverses_overall_validation_advantage": (
            overall_mae_difference < 0 and difference > 0
        )
        or (overall_mae_difference > 0 and difference < 0),
        "selected_absolute_error_fractions": {
            "below_1_yard": float((absolute < 1).mean()),
            "at_least_3_yards": float((absolute >= 3).mean()),
            "at_least_5_yards": float((absolute >= 5).mean()),
        },
    }


def _classification_metrics(
    keys: pd.DataFrame, target: np.ndarray, probability: np.ndarray
) -> dict[str, float | int | None]:
    labels = target.astype(np.int8)
    return {
        "sample_count": int(len(labels)),
        "play_count": _play_count(keys),
        "observed_closing_rate": float(labels.mean()),
        "mean_predicted_closing_probability": float(probability.mean()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probability)),
        "roc_auc": (
            float(roc_auc_score(labels, probability))
            if len(np.unique(labels)) == 2
            else None
        ),
        "accuracy_at_0_5": float(
            accuracy_score(labels, probability >= 0.5)
        ),
        "probability_bias": float(probability.mean() - labels.mean()),
    }


def classification_bucket_diagnostics(
    keys: pd.DataFrame,
    target: Sequence[int] | pd.Series | np.ndarray,
    selected_probability: Sequence[float] | pd.Series | np.ndarray,
    comparator_probability: Sequence[float] | pd.Series | np.ndarray,
    *,
    overall_log_loss_difference: float,
) -> dict[str, Any]:
    y, selected, comparator = _arrays(
        target, selected_probability, comparator_probability
    )
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Classification target must be binary")
    if ((selected < 0) | (selected > 1)).any() or (
        (comparator < 0) | (comparator > 1)
    ).any():
        raise ValueError("Classification probabilities must be within [0, 1]")
    labels = y.astype(np.int8)
    selected_metrics = _classification_metrics(
        keys, labels, selected
    )
    comparator_metrics = _classification_metrics(
        keys, labels, comparator
    )
    log_difference = float(
        selected_metrics["log_loss"] - comparator_metrics["log_loss"]
    )
    brier_difference = float(
        selected_metrics["brier_score"] - comparator_metrics["brier_score"]
    )
    predicted = selected >= 0.5
    positive = labels == 1
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    tn = int((~predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    return {
        "selected_model": selected_metrics,
        "comparator": comparator_metrics,
        "log_loss_difference_selected_minus_comparator": log_difference,
        "brier_difference_selected_minus_comparator": brier_difference,
        "selected_beats_comparator": log_difference < 0,
        "reverses_overall_validation_advantage": (
            overall_log_loss_difference < 0 and log_difference > 0
        )
        or (overall_log_loss_difference > 0 and log_difference < 0),
        "contains_both_classes": len(np.unique(labels)) == 2,
        "selected_confusion_at_0_5": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "false_positive_rate": (
                float(fp / (fp + tn)) if fp + tn else None
            ),
            "false_negative_rate": (
                float(fn / (fn + tp)) if fn + tp else None
            ),
        },
    }


def adequate_bucket_bootstrap(
    keys: pd.DataFrame,
    target: Sequence[float] | pd.Series | np.ndarray,
    selected: Sequence[float] | pd.Series | np.ndarray,
    comparator: Sequence[float] | pd.Series | np.ndarray,
    *,
    task: str,
    support: str,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any] | None:
    if support != "adequate":
        return None
    y, chosen, baseline = _arrays(target, selected, comparator)
    play_keys = keys[["game_id", "play_id"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    names = (
        ("mae_difference",)
        if task == "regression"
        else ("log_loss_difference", "brier_difference")
    )
    draws = {name: np.empty(resamples) for name in names}
    for draw in range(resamples):
        positions = bootstrap_play_row_positions(play_keys, rng)
        sampled_y = y[positions]
        sampled_selected = chosen[positions]
        sampled_comparator = baseline[positions]
        if task == "regression":
            draws["mae_difference"][draw] = mean_absolute_error(
                sampled_y, sampled_selected
            ) - mean_absolute_error(sampled_y, sampled_comparator)
        elif task == "classification":
            sampled_labels = sampled_y.astype(np.int8)
            draws["log_loss_difference"][draw] = log_loss(
                sampled_labels, sampled_selected, labels=[0, 1]
            ) - log_loss(
                sampled_labels, sampled_comparator, labels=[0, 1]
            )
            draws["brier_difference"][draw] = brier_score_loss(
                sampled_labels, sampled_selected
            ) - brier_score_loss(sampled_labels, sampled_comparator)
        else:
            raise ValueError(f"Unsupported bootstrap task: {task}")
    if not all(np.isfinite(values).all() for values in draws.values()):
        raise ValueError("Bucket bootstrap produced non-finite values")
    return {
        "seed": int(seed),
        "completed_resamples": int(resamples),
        "sampling_unit": "game_id/play_id within bucket",
        "confidence_level": CONFIDENCE_LEVEL,
        "intervals": {
            name: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for name, values in draws.items()
        },
    }


def empty_bucket_record(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        **bucket,
        "sample_count": 0,
        "play_count": 0,
        "support_flag": "sparse",
        "observed_mean_separation_change": None,
        "diagnostics": None,
        "bootstrap": None,
    }
