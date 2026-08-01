from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from gridiron_spatial.calibration_analysis import (
    brier_decomposition,
    calibration_diagnostics,
    calibration_errors,
    calibration_intercept_slope,
    confidence_regions,
    fixed_reliability_bins,
    play_cluster_bootstrap_intervals,
    validate_calibration_weeks,
)
from gridiron_spatial.model_interpretation import (
    bootstrap_play_row_positions,
)


def _keys(count: int, pairs: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": 10 + np.arange(count) // (5 * pairs),
            "play_id": np.arange(count) // pairs,
        }
    )


def test_fixed_bins_include_endpoints_empty_bins_and_gap_sign():
    keys = _keys(4)
    bins = fixed_reliability_bins(
        keys, [0, 1, 1, 0], [0.0, 0.1, 0.85, 1.0]
    )
    assert len(bins) == 10
    assert bins[0]["sample_count"] == 1
    assert bins[9]["sample_count"] == 1
    assert bins[9]["upper_bound_inclusive"] is True
    assert bins[2]["sample_count"] == 0
    assert bins[2]["mean_predicted_probability"] is None
    assert bins[1]["calibration_gap"] == pytest.approx(0.9)
    errors = calibration_errors(bins, 4)
    expected = sum(
        row["sample_count"] / 4 * abs(row["calibration_gap"])
        for row in bins
        if row["sample_count"]
    )
    assert errors["ece"] == pytest.approx(expected)
    assert errors["mce"] == pytest.approx(
        max(
            abs(row["calibration_gap"])
            for row in bins
            if row["sample_count"]
        )
    )


def test_brier_decomposition_reconstructs_bin_mean_forecast():
    probability = np.repeat([0.2, 0.8], 10)
    target = np.array([1, 1, *([0] * 8), *([1] * 8), 0, 0])
    bins = fixed_reliability_bins(_keys(20), target, probability)
    result = brier_decomposition(target, probability, bins)
    assert result["reliability"] == pytest.approx(0)
    assert result["absolute_reconstruction_error"] <= 1e-10
    assert result["reconstructed_brier_score"] == pytest.approx(
        result["decomposition_reference_brier_score"]
    )


def test_known_calibration_intercept_and_slope_are_near_ideal():
    probability = np.repeat(np.arange(0.1, 1.0, 0.1), 100)
    target = np.concatenate(
        [
            np.r_[np.ones(int(round(value * 100))), np.zeros(100 - int(round(value * 100)))]
            for value in np.arange(0.1, 1.0, 0.1)
        ]
    )
    result = calibration_intercept_slope(target, probability)
    assert result["convergence_status"] == "CONVERGED"
    assert result["intercept"] == pytest.approx(0, abs=0.03)
    assert result["slope"] == pytest.approx(1, abs=0.03)


def test_confidence_regions_keep_040_and_060_in_middle():
    result = confidence_regions(
        _keys(5), [0, 0, 1, 1, 1], [0.39, 0.40, 0.50, 0.60, 0.61]
    )
    assert [row["sample_count"] for row in result] == [1, 3, 1]


def test_play_bootstrap_retains_pair_clusters_and_is_deterministic():
    keys = _keys(20, pairs=2)
    positions = bootstrap_play_row_positions(
        keys, np.random.default_rng(2026)
    )
    selected = keys.iloc[positions]
    assert all(
        count % 2 == 0
        for count in selected.groupby(["game_id", "play_id"]).size()
    )
    target = np.tile([0, 1], 10)
    probability = np.tile([0.3, 0.7], 10)
    first = play_cluster_bootstrap_intervals(
        keys, target, probability, resamples=8
    )
    second = play_cluster_bootstrap_intervals(
        keys, target, probability, resamples=8
    )
    assert first == second
    assert first["sampling_unit"] == "game_id/play_id"


def test_invalid_probabilities_and_targets_fail():
    with pytest.raises(ValueError, match="within"):
        fixed_reliability_bins(_keys(2), [0, 1], [-0.1, 0.5])
    with pytest.raises(ValueError, match="binary"):
        fixed_reliability_bins(_keys(2), [0, 2], [0.2, 0.8])


def test_frozen_week_rejection():
    development = tuple(f"2023_w{week:02d}" for week in range(1, 13))
    validation = ("2023_w13", "2023_w14", "2023_w15")
    assert validate_calibration_weeks(development, validation)
    with pytest.raises(ValueError, match="frozen-test"):
        validate_calibration_weeks(development, (*validation, "2023_w16"))


def test_complete_diagnostics_do_not_mutate_inputs():
    keys = _keys(40, pairs=2)
    target = pd.Series(np.tile([0, 1], 20))
    probability = pd.Series(np.tile([0.25, 0.75], 20))
    keys_before = keys.copy(deep=True)
    target_before = target.copy(deep=True)
    probability_before = probability.copy(deep=True)
    result = calibration_diagnostics(
        keys, target, probability, bootstrap_resamples=8
    )
    assert sum(
        row["sample_count"] for row in result["reliability_bins"]
    ) == 40
    assert sum(
        row["sample_count"] for row in result["confidence_regions"]
    ) == 40
    pdt.assert_frame_equal(keys, keys_before)
    pdt.assert_series_equal(target, target_before)
    pdt.assert_series_equal(probability, probability_before)
