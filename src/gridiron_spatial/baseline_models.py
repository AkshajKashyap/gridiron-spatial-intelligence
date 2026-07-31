"""Pre-registered linear baseline fitting and validation selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_features import (
    REGISTERED_NUMERIC_FEATURES,
    feature_matrix,
)


RIDGE_ALPHAS = (0.1, 1.0, 10.0)
RIDGE_TIE_ORDER = (10.0, 1.0, 0.1)
LOGISTIC_CS = (0.1, 1.0, 10.0)
EXPECTED_WEEKS = tuple(f"2023_w{week:02d}" for week in range(1, 16))
DEVELOPMENT_WEEKS = EXPECTED_WEEKS[:12]
VALIDATION_WEEKS = EXPECTED_WEEKS[12:]
PREPROCESSING_SPECIFICATION = {
    "numeric_imputation": "median fit on development_train only",
    "numeric_scaling": "standardization fit on development_train only",
    "categorical_preprocessing": "none; optional categoricals omitted",
    "pipeline_fit_population": "development_train",
}


def validate_requested_weeks(weeks: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(weeks)
    for week in requested:
        if not isinstance(week, str) or not week.startswith("2023_w"):
            raise ValueError(f"Malformed week: {week!r}")
        try:
            number = int(week.removeprefix("2023_w"))
        except ValueError as error:
            raise ValueError(f"Malformed week: {week!r}") from error
        if number > 15:
            raise ValueError("Frozen-test weeks are prohibited")
    if requested != EXPECTED_WEEKS:
        raise ValueError("Requested weeks must be exactly Weeks 01-15 in order")
    return requested


def build_regression_pipeline(
    candidate: str,
    *,
    alpha: float | None = None,
) -> Pipeline:
    if candidate == "single_feature_linear":
        estimator = LinearRegression()
    elif candidate == "multivariable_ols":
        estimator = LinearRegression()
    elif candidate == "ridge":
        if alpha not in RIDGE_ALPHAS:
            raise ValueError(f"Unregistered ridge alpha: {alpha}")
        estimator = Ridge(alpha=float(alpha))
    else:
        raise ValueError(f"Unsupported regression candidate: {candidate}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )


def build_classification_pipeline(
    candidate: str,
    *,
    c_value: float | None = None,
) -> Pipeline:
    if candidate == "single_feature_logistic":
        c_value = 1.0
    elif candidate == "multivariable_logistic":
        if c_value not in LOGISTIC_CS:
            raise ValueError(f"Unregistered logistic C: {c_value}")
    else:
        raise ValueError(f"Unsupported classification candidate: {candidate}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "estimator",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(y, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(y, estimate))),
        "median_absolute_error": float(median_absolute_error(y, estimate)),
        "r2": float(r2_score(y, estimate)),
    }


def classification_metrics(
    actual: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | None]:
    y = np.asarray(actual, dtype=int)
    estimate = np.asarray(probability, dtype=float)
    classes = np.unique(y)
    auc = float(roc_auc_score(y, estimate)) if len(classes) == 2 else None
    return {
        "log_loss": float(log_loss(y, estimate, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, estimate)),
        "roc_auc": auc,
        "accuracy_at_0_5": float(
            accuracy_score(y, (estimate >= 0.5).astype(int))
        ),
    }


def _record(
    name: str,
    feature_subset: Sequence[str],
    hyperparameter: dict[str, float] | None,
    development_metrics: dict[str, float | None],
    validation_metrics: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "candidate": name,
        "hyperparameter": hyperparameter,
        "feature_subset": list(feature_subset),
        "development_metrics": development_metrics,
        "validation_metrics": validation_metrics,
    }


def _selection(
    candidates: list[dict[str, Any]],
    *,
    primary_metric: str,
    preference: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    priority = {name: rank for rank, name in enumerate(preference)}
    selected = min(
        candidates,
        key=lambda row: (
            row["validation_metrics"][primary_metric],
            priority[row["candidate"]],
        ),
    )
    best = selected["validation_metrics"][primary_metric]
    tied = [
        row["candidate"]
        for row in candidates
        if row["validation_metrics"][primary_metric] == best
    ]
    return selected, {
        "exact_tie": len(tied) > 1,
        "tied_candidates": tied,
        "preference_order": list(preference),
        "selected_after_tie_break": selected["candidate"],
    }


def select_regression_baseline(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = REGISTERED_NUMERIC_FEATURES,
) -> dict[str, Any]:
    """Fit registered regression candidates using development rows only."""

    dev = development.copy(deep=True)
    val = validation.copy(deep=True)
    all_features = tuple(feature_columns)
    x_dev = feature_matrix(dev, all_features)
    x_val = feature_matrix(val, all_features)
    y_dev = pd.to_numeric(dev["separation_change"], errors="raise")
    y_val = pd.to_numeric(val["separation_change"], errors="raise")
    candidates: list[dict[str, Any]] = []

    constants = (
        ("training_mean_constant", float(y_dev.mean())),
        ("training_median_constant", float(y_dev.median())),
    )
    for name, value in constants:
        candidates.append(
            _record(
                name,
                (),
                None,
                regression_metrics(y_dev, np.full(len(dev), value)),
                regression_metrics(y_val, np.full(len(val), value)),
            )
        )
    learned = (
        (
            "single_feature_linear",
            ("separation_origin",),
            build_regression_pipeline("single_feature_linear"),
            None,
        ),
        (
            "multivariable_ols",
            all_features,
            build_regression_pipeline("multivariable_ols"),
            None,
        ),
        *tuple(
            (
                f"ridge_alpha_{alpha:g}",
                all_features,
                build_regression_pipeline("ridge", alpha=alpha),
                {"alpha": alpha},
            )
            for alpha in RIDGE_ALPHAS
        ),
    )
    for name, features, pipeline, hyperparameter in learned:
        pipeline.fit(feature_matrix(dev, features), y_dev)
        candidates.append(
            _record(
                name,
                features,
                hyperparameter,
                regression_metrics(
                    y_dev, pipeline.predict(feature_matrix(dev, features))
                ),
                regression_metrics(
                    y_val, pipeline.predict(feature_matrix(val, features))
                ),
            )
        )
    preference = (
        "training_mean_constant",
        "training_median_constant",
        "single_feature_linear",
        "multivariable_ols",
        "ridge_alpha_10",
        "ridge_alpha_1",
        "ridge_alpha_0.1",
    )
    selected, tie = _selection(
        candidates, primary_metric="mae", preference=preference
    )
    comparator, _ = _selection(
        [
            row
            for row in candidates
            if row["candidate"]
            in {
                "training_mean_constant",
                "training_median_constant",
                "single_feature_linear",
            }
        ],
        primary_metric="mae",
        preference=preference[:3],
    )
    return {
        "task": "regression",
        "primary_metric": "mae",
        "candidates": candidates,
        "selected_candidate": selected["candidate"],
        "selected_hyperparameter": selected["hyperparameter"],
        "selected_feature_subset": selected["feature_subset"],
        "selection_validation_metric": selected["validation_metrics"]["mae"],
        "strongest_constant_or_single_feature": {
            "candidate": comparator["candidate"],
            "validation_mae": comparator["validation_metrics"]["mae"],
        },
        "tie_breaking": tie,
        "preprocessing": PREPROCESSING_SPECIFICATION.copy(),
    }


def select_classification_baseline(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = REGISTERED_NUMERIC_FEATURES,
) -> dict[str, Any]:
    """Fit registered classification candidates using development rows only."""

    dev = development.copy(deep=True)
    val = validation.copy(deep=True)
    all_features = tuple(feature_columns)
    x_dev = feature_matrix(dev, all_features)
    x_val = feature_matrix(val, all_features)
    y_dev = pd.to_numeric(dev["closing"], errors="raise").astype(int)
    y_val = pd.to_numeric(val["closing"], errors="raise").astype(int)
    if y_dev.nunique() != 2:
        raise ValueError("Development classification target needs both classes")
    rate = float(y_dev.mean())
    candidates = [
        _record(
            "training_closing_rate_constant",
            (),
            None,
            classification_metrics(y_dev, np.full(len(dev), rate)),
            classification_metrics(y_val, np.full(len(val), rate)),
        )
    ]
    learned = (
        (
            "single_feature_logistic",
            ("separation_origin",),
            build_classification_pipeline("single_feature_logistic"),
            None,
        ),
        *tuple(
            (
                f"multivariable_logistic_c_{c_value:g}",
                all_features,
                build_classification_pipeline(
                    "multivariable_logistic", c_value=c_value
                ),
                {"C": c_value},
            )
            for c_value in LOGISTIC_CS
        ),
    )
    for name, features, pipeline, hyperparameter in learned:
        pipeline.fit(feature_matrix(dev, features), y_dev)
        candidates.append(
            _record(
                name,
                features,
                hyperparameter,
                classification_metrics(
                    y_dev,
                    pipeline.predict_proba(feature_matrix(dev, features))[:, 1],
                ),
                classification_metrics(
                    y_val,
                    pipeline.predict_proba(feature_matrix(val, features))[:, 1],
                ),
            )
        )
    preference = (
        "training_closing_rate_constant",
        "single_feature_logistic",
        "multivariable_logistic_c_0.1",
        "multivariable_logistic_c_1",
        "multivariable_logistic_c_10",
    )
    selected, tie = _selection(
        candidates, primary_metric="log_loss", preference=preference
    )
    comparator, _ = _selection(
        [
            row
            for row in candidates
            if row["candidate"]
            in {
                "training_closing_rate_constant",
                "single_feature_logistic",
            }
        ],
        primary_metric="log_loss",
        preference=preference[:2],
    )
    return {
        "task": "classification",
        "primary_metric": "log_loss",
        "candidates": candidates,
        "selected_candidate": selected["candidate"],
        "selected_hyperparameter": selected["hyperparameter"],
        "selected_feature_subset": selected["feature_subset"],
        "selection_validation_metric": selected["validation_metrics"][
            "log_loss"
        ],
        "strongest_constant_or_single_feature": {
            "candidate": comparator["candidate"],
            "validation_log_loss": comparator["validation_metrics"][
                "log_loss"
            ],
        },
        "tie_breaking": tie,
        "preprocessing": PREPROCESSING_SPECIFICATION.copy(),
    }
