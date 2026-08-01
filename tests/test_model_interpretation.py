from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from gridiron_spatial.baseline_features import REGISTERED_NUMERIC_FEATURES
from gridiron_spatial.model_interpretation import (
    ABLATION_ORDER,
    bootstrap_play_row_positions,
    coefficient_stability,
    fixed_ablation_evaluation,
    fixed_ablation_feature_sets,
    summarize_coefficients,
    validate_interpretation_weeks,
)


def _samples(plays: int = 30, pairs: int = 2) -> pd.DataFrame:
    play = np.repeat(np.arange(plays), pairs)
    row = np.arange(len(play), dtype=float)
    rng = np.random.default_rng(14)
    frame = pd.DataFrame(
        {
            "game_id": 1000 + play // 5,
            "play_id": play,
            "separation_origin": 1.5 + row / 20,
            "dx": rng.normal(size=len(row)),
            "dy": rng.normal(size=len(row)),
            "target_x_origin": 20 + row / 10,
            "target_y_origin": 10 + row / 30,
            "defender_x_origin": 21 + row / 10,
            "defender_y_origin": 11 + row / 30,
            "valid_observed_defender_count_origin": pairs,
            "defender_rank_origin": np.tile(np.arange(1, pairs + 1), plays),
            "nearest_observed_defender_indicator": np.tile(
                [1, *([0] * (pairs - 1))], plays
            ),
        }
    )
    frame["abs_dx"] = frame["dx"].abs()
    frame["abs_dy"] = frame["dy"].abs()
    frame["separation_change"] = (
        0.8 * frame["dx"] - 0.35 * frame["dy"] + rng.normal(0, 0.15, len(row))
    )
    frame["closing"] = frame["separation_change"].lt(0).astype("int8")
    return frame


def test_fixed_ablation_sets_are_exact_and_add_nothing():
    sets = fixed_ablation_feature_sets()
    assert tuple(sets) == ABLATION_ORDER
    assert sets["full"] == REGISTERED_NUMERIC_FEATURES
    assert sets["separation_only"] == ("separation_origin",)
    assert set(sets["without_field_location"]) == set(
        REGISTERED_NUMERIC_FEATURES
    ) - {
        "target_x_origin",
        "target_y_origin",
        "defender_x_origin",
        "defender_y_origin",
    }
    assert set(sets["without_relative_vector"]) == set(
        REGISTERED_NUMERIC_FEATURES
    ) - {"dx", "dy", "abs_dx", "abs_dy"}
    assert set(sets["without_defender_context"]) == set(
        REGISTERED_NUMERIC_FEATURES
    ) - {
        "valid_observed_defender_count_origin",
        "defender_rank_origin",
        "nearest_observed_defender_indicator",
    }
    assert all(
        set(features).issubset(REGISTERED_NUMERIC_FEATURES)
        for features in sets.values()
    )
    with pytest.raises(ValueError, match="frozen"):
        fixed_ablation_feature_sets((*REGISTERED_NUMERIC_FEATURES, "week"))


def test_week_gate_rejects_frozen_or_inexact_weeks():
    development = tuple(f"2023_w{week:02d}" for week in range(1, 13))
    validation = ("2023_w13", "2023_w14", "2023_w15")
    assert validate_interpretation_weeks(development, validation) == (
        development,
        validation,
    )
    with pytest.raises(ValueError, match="frozen-test"):
        validate_interpretation_weeks(development, (*validation, "2023_w16"))
    with pytest.raises(ValueError, match="requires"):
        validate_interpretation_weeks(development[:-1], validation)


def test_play_bootstrap_is_clustered_deterministic_and_nonmutating():
    samples = _samples(12, 3)
    before = samples.copy(deep=True)
    first = bootstrap_play_row_positions(samples, np.random.default_rng(2026))
    second = bootstrap_play_row_positions(samples, np.random.default_rng(2026))
    assert np.array_equal(first, second)
    selected = samples.iloc[first]
    original_counts = samples.groupby(["game_id", "play_id"]).size()
    selected_counts = selected.groupby(["game_id", "play_id"]).size()
    for key, count in selected_counts.items():
        assert count % original_counts.loc[key] == 0
    pdt.assert_frame_equal(samples, before)


def test_coefficient_summary_and_sign_stability():
    result = summarize_coefficients(
        ("a", "b"),
        np.array([1.0, -2.0]),
        np.array([[1.0, -1.0], [2.0, -3.0], [-1.0, -2.0], [3.0, 4.0]]),
    )
    assert result[0]["positive_fraction"] == 0.75
    assert result[0]["negative_fraction"] == 0.25
    assert result[0]["sign_stability"] == 0.75
    assert result[1]["sign_stability"] == 0.75
    with pytest.raises(ValueError, match="non-finite"):
        summarize_coefficients(
            ("a",), np.array([1.0]), np.array([[np.nan]])
        )


def test_coefficient_bootstrap_is_deterministic_and_development_only():
    development = _samples()
    before = development.copy(deep=True)
    kwargs = {
        "task": "regression",
        "candidate": "multivariable_ols",
        "hyperparameter": None,
        "resamples": 12,
    }
    first = coefficient_stability(development, **kwargs)
    second = coefficient_stability(development, **kwargs)
    assert first == second
    assert first["completed_bootstrap_resamples"] == 12
    assert first["sampling_unit"] == "game_id/play_id"
    assert first["preprocessing_fit_population"] == "development_train"
    pdt.assert_frame_equal(development, before)


def test_regression_ablation_metrics_use_declared_difference_sign():
    development = _samples(40)
    validation = _samples(15)
    dev_before = development.copy(deep=True)
    val_before = validation.copy(deep=True)
    rows = fixed_ablation_evaluation(
        development,
        validation,
        task="regression",
        candidate="multivariable_ols",
        hyperparameter=None,
    )
    full = rows[0]["mae"]
    assert [row["ablation"] for row in rows] == list(ABLATION_ORDER)
    for row in rows:
        assert row["mae_difference_vs_full"] == pytest.approx(
            row["mae"] - full
        )
    pdt.assert_frame_equal(development, dev_before)
    pdt.assert_frame_equal(validation, val_before)


def test_classification_ablation_metrics_use_declared_difference_sign():
    development = _samples(40)
    validation = _samples(15)
    rows = fixed_ablation_evaluation(
        development,
        validation,
        task="classification",
        candidate="multivariable_logistic_c_10",
        hyperparameter={"C": 10.0},
    )
    full_log_loss = rows[0]["log_loss"]
    full_brier = rows[0]["brier_score"]
    for row in rows:
        assert row["log_loss_difference_vs_full"] == pytest.approx(
            row["log_loss"] - full_log_loss
        )
        assert row["brier_difference_vs_full"] == pytest.approx(
            row["brier_score"] - full_brier
        )
