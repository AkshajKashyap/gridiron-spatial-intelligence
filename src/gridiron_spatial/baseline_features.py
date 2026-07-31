"""Leakage-safe origin-only baseline sample construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .receiver_defender_pairs import PAIR_KEY


HORIZONS = (5, 10, 15)
SAMPLE_KEY = tuple(PAIR_KEY)
ORIGIN_PAIR_KEY = (
    "game_id",
    "play_id",
    "target_nfl_id",
    "defender_nfl_id",
    "origin_frame",
)
REGISTERED_NUMERIC_FEATURES = (
    "separation_origin",
    "dx",
    "dy",
    "abs_dx",
    "abs_dy",
    "target_x_origin",
    "target_y_origin",
    "defender_x_origin",
    "defender_y_origin",
    "valid_observed_defender_count_origin",
    "defender_rank_origin",
    "nearest_observed_defender_indicator",
)
SAMPLE_COLUMNS = (
    *SAMPLE_KEY,
    "week",
    "split",
    *REGISTERED_NUMERIC_FEATURES,
    "separation_change",
    "closing",
)


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_feature_subset(columns: Sequence[str]) -> tuple[str, ...]:
    """Reject every field outside the pre-registered origin-only allowlist."""

    selected = tuple(columns)
    if len(selected) != len(set(selected)):
        raise ValueError("Feature list contains duplicates")
    prohibited = [
        column
        for column in selected
        if column not in REGISTERED_NUMERIC_FEATURES
    ]
    if prohibited:
        raise ValueError(f"Unregistered or prohibited features: {prohibited}")
    if not selected:
        raise ValueError("Feature list must not be empty")
    return selected


def feature_matrix(
    samples: pd.DataFrame,
    columns: Sequence[str] = REGISTERED_NUMERIC_FEATURES,
) -> pd.DataFrame:
    """Return an allowlisted feature matrix with no identifiers."""

    selected = validate_feature_subset(columns)
    _require(samples, selected, "samples")
    return samples.loc[:, list(selected)].copy(deep=True)


def build_baseline_samples(
    pairs: pd.DataFrame,
    origin_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Build deterministic origin-only features and horizon-specific targets."""

    _require(
        pairs,
        [
            *SAMPLE_KEY,
            "week",
            "split",
            "separation_change",
        ],
        "pairs",
    )
    _require(
        origin_pairs,
        [
            *ORIGIN_PAIR_KEY,
            "target_x",
            "target_y",
            "defender_x",
            "defender_y",
            "dx",
            "dy",
            "separation_origin",
        ],
        "origin_pairs",
    )
    if pairs.duplicated(list(SAMPLE_KEY)).any():
        raise ValueError("Duplicate sample keys")
    if origin_pairs.duplicated(list(ORIGIN_PAIR_KEY)).any():
        raise ValueError("Duplicate origin pair keys")
    if not set(pd.to_numeric(pairs["horizon"], errors="coerce")).issubset(
        HORIZONS
    ):
        raise ValueError("Unsupported or malformed horizon")

    origins = origin_pairs.copy(deep=True)
    origins["_defender_text"] = origins["defender_nfl_id"].astype("string")
    origins["_defender_number"] = pd.to_numeric(
        origins["defender_nfl_id"], errors="coerce"
    )
    origins["_defender_number_missing"] = origins["_defender_number"].isna()
    origin_group = [
        "game_id",
        "play_id",
        "target_nfl_id",
        "origin_frame",
    ]
    origins["valid_observed_defender_count_origin"] = (
        origins.groupby(origin_group, sort=False)["defender_nfl_id"]
        .transform("nunique")
        .astype("int64")
    )
    origins = origins.sort_values(
        [
            *origin_group,
            "separation_origin",
            "_defender_number_missing",
            "_defender_number",
            "_defender_text",
        ],
        kind="stable",
    )
    origins["defender_rank_origin"] = (
        origins.groupby(origin_group, sort=False).cumcount() + 1
    ).astype("int64")
    origins["nearest_observed_defender_indicator"] = (
        origins["defender_rank_origin"].eq(1).astype("int8")
    )
    origins["abs_dx"] = origins["dx"].abs()
    origins["abs_dy"] = origins["dy"].abs()
    origins = origins.rename(
        columns={
            "target_x": "target_x_origin",
            "target_y": "target_y_origin",
            "defender_x": "defender_x_origin",
            "defender_y": "defender_y_origin",
        }
    )
    origin_features = origins[
        [*ORIGIN_PAIR_KEY, *REGISTERED_NUMERIC_FEATURES]
    ]
    samples = pairs[
        [*SAMPLE_KEY, "week", "split", "separation_change"]
    ].copy(deep=True)
    samples = samples.merge(
        origin_features,
        on=list(ORIGIN_PAIR_KEY),
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if samples["_merge"].ne("both").any():
        raise ValueError("Horizon sample is missing origin-only features")
    samples = samples.drop(columns="_merge")
    numeric = samples[list(REGISTERED_NUMERIC_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    )
    if (~np.isfinite(numeric.to_numpy(dtype=float))).any():
        raise ValueError("Origin-only features contain non-finite values")
    target = pd.to_numeric(samples["separation_change"], errors="coerce")
    if (~np.isfinite(target)).any():
        raise ValueError("Regression target contains non-finite values")
    samples["closing"] = target.lt(0).astype("int8")
    samples = samples.loc[:, list(SAMPLE_COLUMNS)].sort_values(
        list(SAMPLE_KEY), kind="stable"
    )
    samples = samples.reset_index(drop=True)
    if samples.duplicated(list(SAMPLE_KEY)).any():
        raise ValueError("Duplicate sample keys after feature construction")
    return samples


def samples_for_horizon(
    samples: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon}")
    selected = samples.loc[samples["horizon"].eq(horizon)].copy(deep=True)
    return selected.sort_values(list(SAMPLE_KEY), kind="stable").reset_index(
        drop=True
    )
