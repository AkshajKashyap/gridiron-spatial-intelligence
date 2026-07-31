import pandas as pd
import pytest

from gridiron_spatial.baseline_features import (
    REGISTERED_NUMERIC_FEATURES,
    SAMPLE_KEY,
    build_baseline_samples,
    feature_matrix,
    samples_for_horizon,
)


def _tables():
    origins = pd.DataFrame(
        [
            {
                "game_id": "1",
                "play_id": "10",
                "target_nfl_id": "T",
                "defender_nfl_id": defender,
                "origin_frame": 4,
                "target_x": 20.0,
                "target_y": 10.0,
                "defender_x": x,
                "defender_y": y,
                "dx": x - 20.0,
                "dy": y - 10.0,
                "separation_origin": separation,
            }
            for defender, x, y, separation in (
                ("10", 21.0, 10.0, 1.0),
                ("2", 20.0, 11.0, 1.0),
                ("3", 23.0, 14.0, 5.0),
            )
        ]
    )
    rows = []
    for horizon in (5, 10, 15):
        for defender, change in (("10", -1.0), ("2", 0.0), ("3", 2.0)):
            rows.append(
                {
                    "game_id": "1",
                    "play_id": "10",
                    "target_nfl_id": "T",
                    "defender_nfl_id": defender,
                    "origin_frame": 4,
                    "horizon": horizon,
                    "week": "2023_w01",
                    "split": "development_train",
                    "separation_change": change,
                }
            )
    return pd.DataFrame(rows), origins


def test_registered_origin_features_rank_count_targets_and_nonmutation():
    pairs, origins = _tables()
    pairs_before = pairs.copy(deep=True)
    origins_before = origins.copy(deep=True)
    samples = build_baseline_samples(pairs, origins)

    assert list(feature_matrix(samples).columns) == list(
        REGISTERED_NUMERIC_FEATURES
    )
    defender_2 = samples.loc[
        samples["defender_nfl_id"].eq("2") & samples["horizon"].eq(5)
    ].iloc[0]
    defender_10 = samples.loc[
        samples["defender_nfl_id"].eq("10") & samples["horizon"].eq(5)
    ].iloc[0]
    defender_3 = samples.loc[
        samples["defender_nfl_id"].eq("3") & samples["horizon"].eq(5)
    ].iloc[0]
    assert defender_2["defender_rank_origin"] == 1
    assert defender_10["defender_rank_origin"] == 2
    assert defender_3["defender_rank_origin"] == 3
    assert defender_2["nearest_observed_defender_indicator"] == 1
    assert defender_10["nearest_observed_defender_indicator"] == 0
    assert defender_3["valid_observed_defender_count_origin"] == 3
    assert defender_3["abs_dx"] == 3.0
    assert defender_3["abs_dy"] == 4.0
    assert samples.loc[samples["defender_nfl_id"].eq("10"), "closing"].eq(
        1
    ).all()
    pd.testing.assert_frame_equal(pairs, pairs_before)
    pd.testing.assert_frame_equal(origins, origins_before)


def test_horizons_are_separate_and_order_is_deterministic():
    pairs, origins = _tables()
    samples = build_baseline_samples(pairs.sample(frac=1, random_state=7), origins)
    assert not samples.duplicated(list(SAMPLE_KEY)).any()
    for horizon in (5, 10, 15):
        selected = samples_for_horizon(samples, horizon)
        assert set(selected["horizon"]) == {horizon}
        assert len(selected) == 3
    assert samples.equals(
        samples.sort_values(list(SAMPLE_KEY), kind="stable").reset_index(
            drop=True
        )
    )


def test_duplicate_samples_and_prohibited_features_are_rejected():
    pairs, origins = _tables()
    duplicate = pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate sample"):
        build_baseline_samples(duplicate, origins)

    samples = build_baseline_samples(pairs, origins)
    with pytest.raises(ValueError, match="prohibited"):
        feature_matrix(samples.assign(separation_future=1.0), ["separation_future"])
