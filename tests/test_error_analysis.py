from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from gridiron_spatial.error_analysis import (
    BUCKET_ORDER,
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


def _keys(count: int, pairs: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": 100 + np.arange(count) // (5 * pairs),
            "play_id": np.arange(count) // pairs,
        }
    )


def test_exact_bucket_boundaries_and_empty_preservation():
    values = [0, 2.999, 3, 4.999, 5, 9.999, 10, 15, 20, 100]
    assigned = assign_separation_buckets(values)
    assert assigned.tolist() == [
        "0_to_3",
        "0_to_3",
        "3_to_5",
        "3_to_5",
        "5_to_10",
        "5_to_10",
        "10_to_15",
        "15_to_20",
        "20_plus",
        "20_plus",
    ]
    records = [
        empty_bucket_record(bucket)
        for bucket in BUCKETS
        if bucket["bucket"] != "0_to_3"
    ]
    assert tuple(row["bucket"] for row in records) == BUCKET_ORDER[1:]
    assert all(row["support_flag"] == "sparse" for row in records)


def test_negative_and_nonfinite_separation_rejected():
    for values in ([-0.1], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            assign_separation_buckets(values)


def test_regression_conventions_and_input_immutability():
    keys = _keys(4)
    target = pd.Series([0.0, 1.0, 2.0, 3.0])
    selected = pd.Series([1.0, 1.0, 1.0, 2.0])
    comparator = pd.Series([2.0, 2.0, 2.0, 2.0])
    copies = [frame.copy(deep=True) for frame in (keys, target, selected, comparator)]
    result = regression_bucket_diagnostics(
        keys,
        target,
        selected,
        comparator,
        overall_mae_difference=-0.2,
    )
    assert result["selected_model"]["mean_signed_error"] == pytest.approx(-0.25)
    assert result["mae_difference_selected_minus_comparator"] == pytest.approx(
        result["selected_model"]["mae"] - result["comparator"]["mae"]
    )
    pdt.assert_frame_equal(keys, copies[0])
    pdt.assert_series_equal(target, copies[1])
    pdt.assert_series_equal(selected, copies[2])
    pdt.assert_series_equal(comparator, copies[3])


def test_classification_bias_differences_and_confusion():
    result = classification_bucket_diagnostics(
        _keys(4),
        [1, 1, 0, 0],
        [0.8, 0.4, 0.7, 0.2],
        [0.7, 0.6, 0.4, 0.3],
        overall_log_loss_difference=-0.1,
    )
    selected = result["selected_model"]
    comparator = result["comparator"]
    assert selected["probability_bias"] == pytest.approx(0.025)
    assert result["log_loss_difference_selected_minus_comparator"] == pytest.approx(
        selected["log_loss"] - comparator["log_loss"]
    )
    assert result["brier_difference_selected_minus_comparator"] == pytest.approx(
        selected["brier_score"] - comparator["brier_score"]
    )
    assert result["selected_confusion_at_0_5"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
        "false_positive_rate": 0.5,
        "false_negative_rate": 0.5,
    }


def test_support_flag_boundaries():
    assert support_flag(500, 150) == "adequate"
    assert support_flag(499, 150) == "limited"
    assert support_flag(100, 40) == "limited"
    assert support_flag(99, 40) == "sparse"


def test_bootstrap_is_play_clustered_deterministic_and_adequate_only():
    keys = _keys(400, pairs=2)
    target = np.tile([0.0, 1.0], 200)
    selected = target + np.tile([0.2, -0.2], 200)
    comparator = target + np.tile([0.4, -0.4], 200)
    first = adequate_bucket_bootstrap(
        keys,
        target,
        selected,
        comparator,
        task="regression",
        support="adequate",
        resamples=10,
    )
    second = adequate_bucket_bootstrap(
        keys,
        target,
        selected,
        comparator,
        task="regression",
        support="adequate",
        resamples=10,
    )
    assert first == second
    assert first["sampling_unit"] == "game_id/play_id within bucket"
    assert adequate_bucket_bootstrap(
        keys,
        target,
        selected,
        comparator,
        task="regression",
        support="limited",
        resamples=10,
    ) is None


def test_binding_comparator_is_preserved():
    assert (
        validate_binding_comparator("regression", "single_feature_linear")
        == "single_feature_linear"
    )
    assert (
        validate_binding_comparator(
            "classification", "single_feature_logistic"
        )
        == "single_feature_logistic"
    )
    with pytest.raises(ValueError, match="Unsupported"):
        validate_binding_comparator("regression", "new_model")


def test_frozen_week_rejection():
    development = tuple(f"2023_w{week:02d}" for week in range(1, 13))
    validation = ("2023_w13", "2023_w14", "2023_w15")
    assert validate_error_analysis_weeks(development, validation)
    with pytest.raises(ValueError, match="frozen-test"):
        validate_error_analysis_weeks(development, (*validation, "2023_w16"))
