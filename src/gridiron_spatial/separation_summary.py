"""Fixed descriptive summaries for receiver-observed-defender separation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .receiver_defender_pairs import PAIR_KEY


HORIZONS = (5, 10, 15)
SPLITS = ("development_train", "validation", "frozen_test")
ORIGIN_SEPARATION_BUCKETS = ("[0,3)", "[3,5)", "[5,10)", "[10,15)", "[15,20)", "[20,inf)")
DEFENDER_COUNT_BUCKETS = ("1-2", "3-4", "5+")
BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 500
_ORIGIN_KEY = ["game_id", "play_id", "target_nfl_id", "origin_frame"]
_PLAY_KEY = ["game_id", "play_id"]


def _validate_distances(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if (~np.isfinite(values) | values.lt(0)).any():
            raise ValueError(f"{column} contains negative or non-finite values")


def prepare_analysis_pairs(
    pairs: pd.DataFrame,
    origin_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Add fixed buckets and deterministic nearest-origin membership."""

    if pairs.duplicated(PAIR_KEY).any():
        raise ValueError("Duplicate horizon pair identities")
    origin_pair_key = [*_ORIGIN_KEY, "defender_nfl_id"]
    if origin_pairs.duplicated(origin_pair_key).any():
        raise ValueError("Duplicate origin pair identities")
    _validate_distances(
        pairs, ["separation_origin", "separation_future"]
    )
    _validate_distances(origin_pairs, ["separation_origin"])

    counts = (
        origin_pairs.groupby(_ORIGIN_KEY, sort=False)["defender_nfl_id"]
        .nunique()
        .rename("origin_defender_count")
        .reset_index()
    )
    nearest = (
        origin_pairs.assign(
            _defender_sort=origin_pairs["defender_nfl_id"].astype("string"),
            _defender_number=pd.to_numeric(
                origin_pairs["defender_nfl_id"], errors="coerce"
            ),
        )
        .assign(_defender_number_missing=lambda value: value["_defender_number"].isna())
        .sort_values(
            [
                *_ORIGIN_KEY,
                "separation_origin",
                "_defender_number_missing",
                "_defender_number",
                "_defender_sort",
            ],
            kind="stable",
        )
        .drop_duplicates(_ORIGIN_KEY)
        [[*_ORIGIN_KEY, "defender_nfl_id"]]
        .assign(is_nearest_observed_defender=True)
    )
    result = pairs.copy(deep=True).merge(
        counts, on=_ORIGIN_KEY, how="left", validate="many_to_one"
    )
    if result["origin_defender_count"].isna().any():
        raise ValueError("Horizon pair is missing its origin defender universe")
    result = result.merge(
        nearest,
        on=[*_ORIGIN_KEY, "defender_nfl_id"],
        how="left",
        validate="many_to_one",
    )
    result["is_nearest_observed_defender"] = (
        result["is_nearest_observed_defender"].fillna(False).astype(bool)
    )
    result["origin_separation_bucket"] = pd.cut(
        result["separation_origin"],
        bins=[0.0, 3.0, 5.0, 10.0, 15.0, 20.0, np.inf],
        labels=ORIGIN_SEPARATION_BUCKETS,
        right=False,
        include_lowest=True,
    ).astype("string")
    result["origin_defender_count_bucket"] = pd.cut(
        result["origin_defender_count"],
        bins=[0, 2, 4, np.inf],
        labels=DEFENDER_COUNT_BUCKETS,
        right=True,
        include_lowest=True,
    ).astype("string")
    if (
        result["origin_separation_bucket"].isna().any()
        or result["origin_defender_count_bucket"].isna().any()
    ):
        raise ValueError("Fixed bucket assignment failed")
    return result


def separation_metrics(frame: pd.DataFrame) -> dict[str, int | float]:
    """Calculate the fixed pair-level descriptive metric set."""

    return separation_metrics_from_arrays(
        frame["separation_origin"].to_numpy(),
        frame["separation_future"].to_numpy(),
        frame["separation_change"].to_numpy(),
        unique_play_count=int(frame[_PLAY_KEY].drop_duplicates().shape[0]),
    )


def separation_metrics_from_arrays(
    separation_origin: np.ndarray,
    separation_future: np.ndarray,
    separation_change: np.ndarray,
    *,
    unique_play_count: int,
) -> dict[str, int | float]:
    """Calculate metrics from identity-free value vectors."""

    origin = np.asarray(separation_origin, dtype=float)
    future = np.asarray(separation_future, dtype=float)
    change = np.asarray(separation_change, dtype=float)
    if not (len(origin) == len(future) == len(change)):
        raise ValueError("Separation metric vectors have inconsistent lengths")
    if not len(change):
        return {
            "pair_count": 0,
            "unique_play_count": 0,
            **{
                name: float("nan")
                for name in (
                    "mean_separation_origin",
                    "median_separation_origin",
                    "mean_separation_future",
                    "median_separation_future",
                    "mean_separation_change",
                    "median_separation_change",
                    "p10_separation_change",
                    "p25_separation_change",
                    "p75_separation_change",
                    "p90_separation_change",
                    "closing_fraction",
                    "unchanged_fraction",
                    "expanding_fraction",
                )
            },
        }
    return {
        "pair_count": int(len(change)),
        "unique_play_count": int(unique_play_count),
        "mean_separation_origin": float(np.mean(origin)),
        "median_separation_origin": float(np.median(origin)),
        "mean_separation_future": float(np.mean(future)),
        "median_separation_future": float(np.median(future)),
        "mean_separation_change": float(np.mean(change)),
        "median_separation_change": float(np.median(change)),
        "p10_separation_change": float(np.quantile(change, 0.10)),
        "p25_separation_change": float(np.quantile(change, 0.25)),
        "p75_separation_change": float(np.quantile(change, 0.75)),
        "p90_separation_change": float(np.quantile(change, 0.90)),
        "closing_fraction": float(np.mean(change < 0)),
        "unchanged_fraction": float(np.mean(change == 0)),
        "expanding_fraction": float(np.mean(change > 0)),
    }


def grouped_separation_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Return deterministic grouped metric records."""

    records = []
    for group, selected in frame.groupby(
        list(group_columns), sort=True, observed=True
    ):
        values = group if isinstance(group, tuple) else (group,)
        records.append(
            {
                **{
                    column: value.item()
                    if isinstance(value, np.generic)
                    else value
                    for column, value in zip(group_columns, values)
                },
                **separation_metrics(selected),
            }
        )
    return records


def play_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, int | float | list[float]]:
    """Bootstrap equal-weighted play aggregates, never individual pairs."""

    play = (
        frame.assign(_closing=frame["separation_change"].lt(0).astype(float))
        .groupby(_PLAY_KEY, sort=True)
        .agg(
            mean_separation_change=("separation_change", "mean"),
            closing_fraction=("_closing", "mean"),
        )
        .reset_index()
    )
    return play_aggregate_bootstrap(
        play,
        seed=seed,
        resamples=resamples,
    )


def play_aggregate_bootstrap(
    play: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, int | float | list[float]]:
    """Bootstrap pre-aggregated, equal-weighted play summaries."""

    if play.empty:
        raise ValueError("Cannot bootstrap an empty play set")
    values = play[["mean_separation_change", "closing_fraction"]].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(play), size=(resamples, len(play)))
    estimates = values[indices].mean(axis=1)
    intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "play_count": int(len(play)),
        "play_weighted_mean_separation_change": float(values[:, 0].mean()),
        "mean_separation_change_95_interval": [
            float(intervals[0, 0]),
            float(intervals[1, 0]),
        ],
        "play_weighted_closing_fraction": float(values[:, 1].mean()),
        "closing_fraction_95_interval": [
            float(intervals[0, 1]),
            float(intervals[1, 1]),
        ],
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
    }


def bootstrap_by_split_horizon(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = []
    for split in SPLITS:
        for horizon in HORIZONS:
            selected = frame.loc[
                frame["split"].astype("string").eq(split)
                & frame["horizon"].eq(horizon)
            ]
            if selected.empty:
                raise ValueError(f"Empty bootstrap group: {split}/{horizon}")
            records.append(
                {
                    "split": split,
                    "horizon": horizon,
                    **play_cluster_bootstrap(selected),
                }
            )
    return records


def validate_weekly_pair_counts(
    frame: pd.DataFrame,
    weekly_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    observed = {
        week: {
            str(horizon): int(
                (
                    frame["week"].astype("string").eq(week)
                    & frame["horizon"].eq(horizon)
                ).sum()
            )
            for horizon in HORIZONS
        }
        for week in weekly_counts
    }
    return {
        "status": "PASS" if observed == weekly_counts else "FAIL",
        "expected": weekly_counts,
        "observed": observed,
    }
