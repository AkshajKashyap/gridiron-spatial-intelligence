from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gridiron_spatial.separation_summary import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFENDER_COUNT_BUCKETS,
    HORIZONS,
    ORIGIN_SEPARATION_BUCKETS,
    bootstrap_by_split_horizon,
    grouped_separation_summary,
    play_cluster_bootstrap,
    prepare_analysis_pairs,
    separation_metrics,
    validate_weekly_pair_counts,
)


def _synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    origin_records = []
    specifications = (
        (10, 100, [(1, 2.0), (2, 4.0)]),
        (20, 200, [(3, 6.0), (4, 6.0), (5, 12.0)]),
        (30, 300, [(6, 20.0), (7, 21.0), (8, 22.0), (9, 23.0), (10, 24.0)]),
        (40, 400, [(11, 15.0)]),
    )
    for play_id, target_id, defenders in specifications:
        for defender_id, separation in defenders:
            origin_records.append(
                {
                    "game_id": 1 if play_id < 30 else 2,
                    "play_id": play_id,
                    "week": "2023_w01" if play_id < 30 else "2023_w02",
                    "split": (
                        "development_train"
                        if play_id < 30
                        else "validation"
                    ),
                    "target_nfl_id": target_id,
                    "defender_nfl_id": defender_id,
                    "origin_frame": 2,
                    "separation_origin": separation,
                }
            )
    origin_pairs = pd.DataFrame(origin_records)
    pair_specs = (
        (10, 100, 1, 5, 2.0, 1.0),
        (10, 100, 2, 5, 4.0, 4.0),
        (10, 100, 2, 10, 4.0, 5.0),
        (20, 200, 3, 5, 6.0, 7.0),
        (20, 200, 4, 5, 6.0, 5.0),
        (20, 200, 5, 10, 12.0, 12.0),
        (30, 300, 6, 15, 20.0, 18.0),
        (30, 300, 7, 15, 21.0, 24.0),
        (40, 400, 11, 5, 15.0, 15.0),
    )
    pair_records = []
    for play_id, target_id, defender_id, horizon, origin, future in pair_specs:
        pair_records.append(
            {
                "game_id": 1 if play_id < 30 else 2,
                "play_id": play_id,
                "week": "2023_w01" if play_id < 30 else "2023_w02",
                "split": (
                    "development_train"
                    if play_id < 30
                    else "validation"
                ),
                "target_nfl_id": target_id,
                "defender_nfl_id": defender_id,
                "origin_frame": 2,
                "horizon": horizon,
                "separation_origin": origin,
                "separation_future": future,
                "separation_change": future - origin,
            }
        )
    return pd.DataFrame(pair_records), origin_pairs


def _record_map(
    records: list[dict[str, object]], *keys: str
) -> dict[tuple[object, ...], dict[str, object]]:
    return {tuple(record[key] for key in keys): record for record in records}


def test_fixed_groups_nearest_selection_and_no_input_mutation() -> None:
    pairs, origins = _synthetic_tables()
    pairs_before = pairs.copy(deep=True)
    origins_before = origins.copy(deep=True)

    prepared = prepare_analysis_pairs(pairs, origins)

    assert tuple(sorted(prepared["horizon"].unique())) == HORIZONS
    assert set(prepared["origin_separation_bucket"]) == set(
        ORIGIN_SEPARATION_BUCKETS
    )
    assert set(prepared["origin_defender_count_bucket"]) == set(
        DEFENDER_COUNT_BUCKETS
    )
    nearest = prepared.loc[prepared["is_nearest_observed_defender"]]
    assert set(zip(nearest["play_id"], nearest["defender_nfl_id"])) == {
        (10, 1),
        (20, 3),
        (30, 6),
        (40, 11),
    }
    assert len(nearest) < len(prepared)
    assert_frame_equal(pairs, pairs_before)
    assert_frame_equal(origins, origins_before)


def test_metrics_horizon_split_week_and_evaluable_origin_population() -> None:
    pairs, origins = _synthetic_tables()
    prepared = prepare_analysis_pairs(pairs, origins)
    metrics = separation_metrics(prepared)
    assert metrics["closing_fraction"] == pytest.approx(3 / 9)
    assert metrics["unchanged_fraction"] == pytest.approx(3 / 9)
    assert metrics["expanding_fraction"] == pytest.approx(3 / 9)

    horizon = _record_map(
        grouped_separation_summary(prepared, ["horizon"]), "horizon"
    )
    assert set(horizon) == {(5,), (10,), (15,)}
    assert horizon[(10,)]["pair_count"] == 2
    assert horizon[(10,)]["mean_separation_origin"] == pytest.approx(8.0)
    split = _record_map(
        grouped_separation_summary(prepared, ["split", "horizon"]),
        "split",
        "horizon",
    )
    assert ("development_train", 5) in split
    assert ("validation", 15) in split
    week = _record_map(
        grouped_separation_summary(prepared, ["week", "horizon"]),
        "week",
        "horizon",
    )
    assert ("2023_w01", 10) in week
    assert ("2023_w02", 15) in week


def test_play_cluster_bootstrap_is_deterministic_and_play_weighted() -> None:
    many = pd.DataFrame(
        {
            "game_id": np.r_[np.repeat(1, 100), 2],
            "play_id": np.r_[np.repeat(10, 100), 20],
            "separation_change": np.r_[np.repeat(-1.0, 100), 3.0],
        }
    )
    first = play_cluster_bootstrap(many)
    second = play_cluster_bootstrap(many)
    assert first == second
    assert first["play_count"] == 2
    assert first["play_weighted_mean_separation_change"] == pytest.approx(1.0)
    assert first["play_weighted_closing_fraction"] == pytest.approx(0.5)
    assert first["bootstrap_seed"] == BOOTSTRAP_SEED
    assert first["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES

    pairs, origins = _synthetic_tables()
    prepared = prepare_analysis_pairs(pairs, origins)
    complete = pd.concat(
        [
            prepared.assign(split=split, horizon=horizon)
            for split in (
                "development_train",
                "validation",
                "frozen_test",
            )
            for horizon in HORIZONS
        ],
        ignore_index=True,
    )
    records = bootstrap_by_split_horizon(complete)
    assert len(records) == 9
    assert [(row["split"], row["horizon"]) for row in records] == [
        (split, horizon)
        for split in (
            "development_train",
            "validation",
            "frozen_test",
        )
        for horizon in HORIZONS
    ]


def test_weekly_reconciliation_and_nonfinite_rejection() -> None:
    pairs, origins = _synthetic_tables()
    prepared = prepare_analysis_pairs(pairs, origins)
    expected = {
        "2023_w01": {"5": 4, "10": 2, "15": 0},
        "2023_w02": {"5": 1, "10": 0, "15": 2},
    }
    assert validate_weekly_pair_counts(prepared, expected)["status"] == "PASS"
    wrong = {**expected, "2023_w02": {"5": 1, "10": 1, "15": 2}}
    assert validate_weekly_pair_counts(prepared, wrong)["status"] == "FAIL"

    invalid = pairs.copy(deep=True)
    invalid.loc[0, "separation_future"] = np.inf
    with pytest.raises(ValueError, match="negative or non-finite"):
        prepare_analysis_pairs(invalid, origins)
