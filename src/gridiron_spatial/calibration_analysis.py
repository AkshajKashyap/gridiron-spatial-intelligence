"""Fixed validation-calibration diagnostics for frozen classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .model_interpretation import (
    bootstrap_play_row_positions,
    validate_interpretation_weeks,
)


BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 500
CONFIDENCE_LEVEL = 0.95
LOGIT_EPSILON = 1e-6
BRIER_RECONSTRUCTION_TOLERANCE = 1e-10
BIN_EDGES = tuple(float(value) / 10 for value in range(11))
REGION_ORDER = ("low", "middle", "high")


def validate_calibration_weeks(
    development_weeks: Sequence[str],
    validation_weeks: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return validate_interpretation_weeks(
        development_weeks, validation_weeks
    )


def _validated_arrays(
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(target)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Target and probability must be nonempty 1D arrays")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    numeric_y = pd.to_numeric(pd.Series(y), errors="raise").to_numpy()
    if not np.isfinite(numeric_y.astype(float)).all() or not set(
        np.unique(numeric_y)
    ).issubset({0, 1}):
        raise ValueError("Calibration target must be binary")
    return numeric_y.astype(np.int8), p


def _bin_ids(probability: np.ndarray) -> np.ndarray:
    return np.minimum((probability * 10).astype(np.int64), 9)


def fixed_reliability_bins(
    sample_keys: pd.DataFrame,
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
) -> list[dict[str, Any]]:
    """Return all ten equal-width bins, including empty bins."""

    y, p = _validated_arrays(target, probability)
    if len(sample_keys) != len(y) or not {
        "game_id",
        "play_id",
    }.issubset(sample_keys):
        raise ValueError("Sample keys must contain aligned game/play fields")
    ids = _bin_ids(p)
    result: list[dict[str, Any]] = []
    for index in range(10):
        selected = ids == index
        count = int(selected.sum())
        mean_probability = float(p[selected].mean()) if count else None
        observed = float(y[selected].mean()) if count else None
        result.append(
            {
                "bin_index": index,
                "lower_bound": BIN_EDGES[index],
                "upper_bound": BIN_EDGES[index + 1],
                "upper_bound_inclusive": index == 9,
                "sample_count": count,
                "play_count": int(
                    sample_keys.loc[selected, ["game_id", "play_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "mean_predicted_probability": mean_probability,
                "observed_closing_fraction": observed,
                "calibration_gap": (
                    None
                    if not count
                    else float(observed - mean_probability)
                ),
            }
        )
    return result


def calibration_errors(
    bins: Sequence[dict[str, Any]], total_count: int
) -> dict[str, float]:
    if total_count <= 0 or sum(row["sample_count"] for row in bins) != total_count:
        raise ValueError("Reliability-bin counts do not reconcile")
    nonempty = [row for row in bins if row["sample_count"]]
    ece = sum(
        row["sample_count"] / total_count * abs(row["calibration_gap"])
        for row in nonempty
    )
    return {
        "ece": float(ece),
        "mce": float(max(abs(row["calibration_gap"]) for row in nonempty)),
    }


def calibration_intercept_slope(
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float | str]:
    """Fit a diagnostic, unregularized logistic calibration equation."""

    y, p = _validated_arrays(target, probability)
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration slope requires both target classes")
    logits = np.log(
        np.clip(p, LOGIT_EPSILON, 1 - LOGIT_EPSILON)
        / (1 - np.clip(p, LOGIT_EPSILON, 1 - LOGIT_EPSILON))
    ).reshape(-1, 1)
    model = LogisticRegression(
        C=np.inf, solver="lbfgs", max_iter=2000, fit_intercept=True
    )
    model.fit(logits, y)
    converged = int(model.n_iter_[0]) < int(model.max_iter)
    if not converged:
        raise ValueError("Calibration intercept/slope fit did not converge")
    values = np.array([model.intercept_[0], model.coef_[0, 0]])
    if not np.isfinite(values).all():
        raise ValueError("Calibration intercept/slope is non-finite")
    return {
        "intercept": float(values[0]),
        "slope": float(values[1]),
        "convergence_status": "CONVERGED",
        "logit_clip_epsilon": LOGIT_EPSILON,
    }


def brier_decomposition(
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
    bins: Sequence[dict[str, Any]],
) -> dict[str, float | str]:
    """Return the standard Murphy decomposition for bin-mean forecasts."""

    y, p = _validated_arrays(target, probability)
    total = len(y)
    observed_rate = float(y.mean())
    reliability = 0.0
    resolution = 0.0
    binned_probability = np.empty(total, dtype=float)
    ids = _bin_ids(p)
    for row in bins:
        if not row["sample_count"]:
            continue
        weight = row["sample_count"] / total
        mean_probability = row["mean_predicted_probability"]
        observed = row["observed_closing_fraction"]
        reliability += weight * (mean_probability - observed) ** 2
        resolution += weight * (observed - observed_rate) ** 2
        binned_probability[ids == row["bin_index"]] = mean_probability
    uncertainty = observed_rate * (1 - observed_rate)
    reconstructed = reliability - resolution + uncertainty
    binned_brier = float(np.mean((binned_probability - y) ** 2))
    error = abs(reconstructed - binned_brier)
    if error > BRIER_RECONSTRUCTION_TOLERANCE:
        raise ValueError("Brier decomposition failed reconstruction tolerance")
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "reconstructed_brier_score": float(reconstructed),
        "decomposition_reference_brier_score": binned_brier,
        "absolute_reconstruction_error": float(error),
        "reconstruction_tolerance": BRIER_RECONSTRUCTION_TOLERANCE,
        "decomposition_reference": "fixed_bin_mean_probabilities",
        "original_probability_brier_score": float(
            brier_score_loss(y, p)
        ),
        "absolute_binning_approximation": float(
            abs(binned_brier - brier_score_loss(y, p))
        ),
    }


def _finite_region_log_loss(y: np.ndarray, p: np.ndarray) -> float | None:
    if ((y == 1) & (p == 0)).any() or ((y == 0) & (p == 1)).any():
        return None
    terms = np.where(y == 1, -np.log(p), -np.log1p(-p))
    return float(terms.mean()) if np.isfinite(terms).all() else None


def confidence_regions(
    sample_keys: pd.DataFrame,
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
) -> list[dict[str, Any]]:
    y, p = _validated_arrays(target, probability)
    masks = {
        "low": p < 0.40,
        "middle": (p >= 0.40) & (p <= 0.60),
        "high": p > 0.60,
    }
    result = []
    for name in REGION_ORDER:
        selected = masks[name]
        count = int(selected.sum())
        result.append(
            {
                "region": name,
                "definition": {
                    "low": "p < 0.40",
                    "middle": "0.40 <= p <= 0.60",
                    "high": "p > 0.60",
                }[name],
                "sample_count": count,
                "play_count": int(
                    sample_keys.loc[selected, ["game_id", "play_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "mean_predicted_probability": (
                    float(p[selected].mean()) if count else None
                ),
                "observed_closing_fraction": (
                    float(y[selected].mean()) if count else None
                ),
                "brier_score": (
                    float(brier_score_loss(y[selected], p[selected]))
                    if count
                    else None
                ),
                "log_loss": (
                    _finite_region_log_loss(y[selected], p[selected])
                    if count
                    else None
                ),
            }
        )
    if sum(row["sample_count"] for row in result) != len(y):
        raise ValueError("Confidence-region counts do not reconcile")
    return result


def predictive_metrics(
    sample_keys: pd.DataFrame,
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float | int | None]:
    y, p = _validated_arrays(target, probability)
    return {
        "sample_count": int(len(y)),
        "play_count": int(
            sample_keys[["game_id", "play_id"]].drop_duplicates().shape[0]
        ),
        "observed_closing_rate": float(y.mean()),
        "mean_predicted_closing_probability": float(p.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else None
        ),
        "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
    }


def _ece(y: np.ndarray, p: np.ndarray) -> float:
    ids = _bin_ids(p)
    counts = np.bincount(ids, minlength=10)
    predicted = np.bincount(ids, weights=p, minlength=10)
    observed = np.bincount(ids, weights=y, minlength=10)
    nonempty = counts > 0
    gaps = observed[nonempty] / counts[nonempty] - (
        predicted[nonempty] / counts[nonempty]
    )
    return float(np.sum(counts[nonempty] / len(y) * np.abs(gaps)))


def play_cluster_bootstrap_intervals(
    sample_keys: pd.DataFrame,
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap fixed validation predictions by game/play cluster."""

    y, p = _validated_arrays(target, probability)
    keys = sample_keys[["game_id", "play_id"]].copy(deep=True).reset_index(
        drop=True
    )
    if len(keys) != len(y) or resamples <= 0:
        raise ValueError("Invalid bootstrap inputs")
    rng = np.random.default_rng(seed)
    draws = {
        "ece": np.empty(resamples),
        "brier_score": np.empty(resamples),
        "mean_probability_minus_observed_rate": np.empty(resamples),
        "calibration_slope": np.empty(resamples),
    }
    for draw in range(resamples):
        positions = bootstrap_play_row_positions(keys, rng)
        sampled_y = y[positions]
        sampled_p = p[positions]
        draws["ece"][draw] = _ece(sampled_y, sampled_p)
        draws["brier_score"][draw] = brier_score_loss(
            sampled_y, sampled_p
        )
        draws["mean_probability_minus_observed_rate"][draw] = (
            sampled_p.mean() - sampled_y.mean()
        )
        draws["calibration_slope"][draw] = calibration_intercept_slope(
            sampled_y, sampled_p
        )["slope"]
    if not all(np.isfinite(values).all() for values in draws.values()):
        raise ValueError("Bootstrap produced non-finite diagnostics")
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return {
        "seed": int(seed),
        "completed_resamples": int(resamples),
        "sampling_unit": "game_id/play_id",
        "confidence_level": CONFIDENCE_LEVEL,
        "intervals": {
            name: {
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1 - alpha)),
            }
            for name, values in draws.items()
        },
    }


def calibration_diagnostics(
    sample_keys: pd.DataFrame,
    target: Sequence[int] | np.ndarray | pd.Series,
    probability: Sequence[float] | np.ndarray | pd.Series,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Build all fixed diagnostics without changing supplied probabilities."""

    keys_before = sample_keys.copy(deep=True)
    y, p = _validated_arrays(target, probability)
    bins = fixed_reliability_bins(sample_keys, y, p)
    result = {
        "predictive_metrics": predictive_metrics(sample_keys, y, p),
        "reliability_bins": bins,
        **calibration_errors(bins, len(y)),
        "calibration_equation": calibration_intercept_slope(y, p),
        "brier_decomposition": brier_decomposition(y, p, bins),
        "confidence_regions": confidence_regions(sample_keys, y, p),
        "play_cluster_bootstrap": play_cluster_bootstrap_intervals(
            sample_keys,
            y,
            p,
            resamples=bootstrap_resamples,
        ),
    }
    if not sample_keys.equals(keys_before):
        raise ValueError("Calibration diagnostics mutated sample keys")
    return result
