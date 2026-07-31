import json

import numpy as np
import pandas as pd
import pytest

from gridiron_spatial.baseline_features import (
    REGISTERED_NUMERIC_FEATURES,
    feature_matrix,
)
from gridiron_spatial.baseline_models import (
    LOGISTIC_CS,
    RIDGE_ALPHAS,
    _selection,
    build_classification_pipeline,
    build_regression_pipeline,
    select_classification_baseline,
    select_regression_baseline,
    validate_requested_weeks,
)


def _samples(count=80, *, week="2023_w01", split="development_train"):
    index = np.arange(count)
    separation = 1.0 + index / 10
    change = np.sin(index / 3) - 0.08 * separation
    frame = pd.DataFrame(
        {
            "game_id": [f"G{value // 4}" for value in index],
            "play_id": [f"P{value}" for value in index],
            "target_nfl_id": "T",
            "defender_nfl_id": [f"D{value}" for value in index],
            "origin_frame": 4,
            "horizon": 5,
            "week": week,
            "split": split,
            "separation_origin": separation,
            "dx": np.cos(index),
            "dy": np.sin(index),
            "abs_dx": np.abs(np.cos(index)),
            "abs_dy": np.abs(np.sin(index)),
            "target_x_origin": 20.0 + index / 20,
            "target_y_origin": 10.0 + index / 30,
            "defender_x_origin": 21.0 + index / 20,
            "defender_y_origin": 11.0 + index / 30,
            "valid_observed_defender_count_origin": 5 + index % 3,
            "defender_rank_origin": 1 + index % 6,
            "nearest_observed_defender_indicator": (index % 6 == 0).astype(
                int
            ),
            "separation_change": change,
            "closing": (change < 0).astype(int),
        }
    )
    return frame


def test_registered_candidates_constants_metrics_and_no_mutation():
    development = _samples()
    validation = _samples(
        40, week="2023_w13", split="validation"
    ).assign(separation_change=lambda value: value["separation_change"] + 0.1)
    validation["closing"] = validation["separation_change"].lt(0).astype(int)
    dev_before = development.copy(deep=True)
    val_before = validation.copy(deep=True)

    regression = select_regression_baseline(development, validation)
    classification = select_classification_baseline(development, validation)

    regression_names = [row["candidate"] for row in regression["candidates"]]
    classification_names = [
        row["candidate"] for row in classification["candidates"]
    ]
    assert regression_names == [
        "training_mean_constant",
        "training_median_constant",
        "single_feature_linear",
        "multivariable_ols",
        "ridge_alpha_0.1",
        "ridge_alpha_1",
        "ridge_alpha_10",
    ]
    assert classification_names == [
        "training_closing_rate_constant",
        "single_feature_logistic",
        "multivariable_logistic_c_0.1",
        "multivariable_logistic_c_1",
        "multivariable_logistic_c_10",
    ]
    assert tuple(
        row["hyperparameter"]["alpha"]
        for row in regression["candidates"]
        if row["hyperparameter"]
    ) == RIDGE_ALPHAS
    assert tuple(
        row["hyperparameter"]["C"]
        for row in classification["candidates"]
        if row["hyperparameter"]
    ) == LOGISTIC_CS
    assert regression["primary_metric"] == "mae"
    assert classification["primary_metric"] == "log_loss"
    constant = classification["candidates"][0]
    expected_rate = development["closing"].mean()
    expected_loss = -(
        validation["closing"] * np.log(expected_rate)
        + (1 - validation["closing"]) * np.log(1 - expected_rate)
    ).mean()
    assert constant["validation_metrics"]["log_loss"] == pytest.approx(
        expected_loss
    )
    json.dumps(
        {"regression": regression, "classification": classification},
        sort_keys=True,
        allow_nan=False,
    )
    pd.testing.assert_frame_equal(development, dev_before)
    pd.testing.assert_frame_equal(validation, val_before)


def test_preprocessing_is_fitted_on_development_only():
    x_development = pd.DataFrame({"separation_origin": [1.0, np.nan, 3.0, 5.0]})
    regression = build_regression_pipeline("single_feature_linear")
    regression.fit(x_development, [0.0, 1.0, 2.0, 3.0])
    assert regression.named_steps["imputer"].statistics_[0] == 3.0

    classification = build_classification_pipeline(
        "single_feature_logistic"
    )
    classification.fit(x_development, [0, 0, 1, 1])
    assert classification.named_steps["imputer"].statistics_[0] == 3.0
    validation = pd.DataFrame({"separation_origin": [1000.0, np.nan]})
    classification.predict_proba(validation)
    assert classification.named_steps["imputer"].statistics_[0] == 3.0


def test_exact_ties_prefer_simpler_and_stronger_ridge():
    candidates = [
        {
            "candidate": name,
            "validation_metrics": {"mae": 1.0},
        }
        for name in (
            "constant",
            "single",
            "ordinary",
            "ridge_alpha_0.1",
            "ridge_alpha_1",
            "ridge_alpha_10",
        )
    ]
    selected, diagnostic = _selection(
        candidates,
        primary_metric="mae",
        preference=(
            "constant",
            "single",
            "ordinary",
            "ridge_alpha_10",
            "ridge_alpha_1",
            "ridge_alpha_0.1",
        ),
    )
    assert selected["candidate"] == "constant"
    assert diagnostic["exact_tie"] is True

    ridge_only = candidates[3:]
    selected, _ = _selection(
        ridge_only,
        primary_metric="mae",
        preference=(
            "ridge_alpha_10",
            "ridge_alpha_1",
            "ridge_alpha_0.1",
        ),
    )
    assert selected["candidate"] == "ridge_alpha_10"


def test_horizon_population_independence_frozen_rejection_and_no_identifier_leakage():
    development = _samples()
    validation = _samples(40, week="2023_w13", split="validation")
    nearest_dev = development.loc[
        development["nearest_observed_defender_indicator"].eq(1)
    ]
    nearest_val = validation.loc[
        validation["nearest_observed_defender_indicator"].eq(1)
    ]
    all_result = select_regression_baseline(development, validation)
    nearest_result = select_regression_baseline(nearest_dev, nearest_val)
    assert all_result is not nearest_result
    h10 = development.assign(horizon=10)
    assert set(h10["horizon"]) == {10}
    assert set(development["horizon"]) == {5}

    valid = [f"2023_w{week:02d}" for week in range(1, 16)]
    assert validate_requested_weeks(valid) == tuple(valid)
    with pytest.raises(ValueError, match="Frozen-test"):
        validate_requested_weeks([*valid[:-1], "2023_w16"])
    with pytest.raises(ValueError, match="prohibited"):
        feature_matrix(development, ["game_id"])
    assert list(feature_matrix(development).columns) == list(
        REGISTERED_NUMERIC_FEATURES
    )
