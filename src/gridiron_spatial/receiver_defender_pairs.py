"""Vectorized target-receiver versus observed-defender separation geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TRACKING_KEY = ["game_id", "play_id", "phase", "frame_id", "nfl_id"]
PAIR_KEY = [
    "game_id",
    "play_id",
    "target_nfl_id",
    "defender_nfl_id",
    "origin_frame",
    "horizon",
]
PAIR_COLUMNS = [
    "game_id",
    "play_id",
    "week",
    "split",
    "target_nfl_id",
    "defender_nfl_id",
    "origin_frame",
    "horizon",
    "target_x_origin",
    "target_y_origin",
    "defender_x_origin",
    "defender_y_origin",
    "dx",
    "dy",
    "separation_origin",
    "target_x_future",
    "target_y_future",
    "defender_x_future",
    "defender_y_future",
    "separation_future",
    "separation_change",
]


@dataclass(frozen=True)
class ReceiverDefenderPairResult:
    pairs: pd.DataFrame
    origin_pairs: pd.DataFrame
    diagnostics: dict[str, Any]


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _explode_defenders(
    frame: pd.DataFrame,
    source_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["defender_nfl_id"] = (
        result[source_column].astype("string").fillna("").str.split("|")
    )
    result = result.explode("defender_nfl_id", ignore_index=True)
    result["defender_nfl_id"] = result["defender_nfl_id"].astype("string")
    return result.loc[result["defender_nfl_id"].ne("")].copy()


def _valid_coordinates(frame: pd.DataFrame, prefix: str) -> pd.Series:
    x = pd.to_numeric(frame[f"{prefix}_x"], errors="coerce")
    y = pd.to_numeric(frame[f"{prefix}_y"], errors="coerce")
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & frame[f"{prefix}_class"].astype("string").ne("invalid")
    )


def build_receiver_defender_pairs(
    tracking: pd.DataFrame,
    primary_origins: pd.DataFrame,
    trajectory_eligibility: pd.DataFrame,
    future_separation_eligibility: pd.DataFrame,
) -> ReceiverDefenderPairResult:
    """Build all eligible target-observed-defender horizon pairs."""

    _require(
        tracking,
        [
            *TRACKING_KEY,
            "week",
            "split",
            "player_side",
            "player_role",
            "x_norm",
            "y_norm",
            "normalized_coordinate_class",
        ],
        "tracking",
    )
    _require(
        primary_origins,
        [
            "game_id",
            "play_id",
            "target_nfl_id",
            "origin_frame",
            "week",
            "split",
            "observed_origin_defender_ids",
            "eligible",
        ],
        "primary_origins",
    )
    _require(
        trajectory_eligibility,
        [
            "game_id",
            "play_id",
            "nfl_id",
            "target_nfl_id",
            "origin_frame",
            "horizon",
            "label_phase",
            "label_end_frame",
            "is_target_role",
            "is_defender",
            "eligible",
        ],
        "trajectory_eligibility",
    )
    _require(
        future_separation_eligibility,
        [
            "game_id",
            "play_id",
            "target_nfl_id",
            "origin_frame",
            "horizon",
            "evaluable_defender_ids",
            "evaluable_defender_count",
            "eligible",
        ],
        "future_separation_eligibility",
    )

    duplicate_rows = tracking.loc[
        tracking.duplicated(TRACKING_KEY, keep=False)
    ]
    if not duplicate_rows.empty:
        if duplicate_rows["player_role"].astype("string").eq(
            "Targeted Receiver"
        ).any():
            raise ValueError("Duplicate target entity-frame rows")
        if (
            duplicate_rows["player_side"].astype("string").eq("Defense")
            | duplicate_rows["player_role"]
            .astype("string")
            .eq("Defensive Coverage")
        ).any():
            raise ValueError("Duplicate defender entity-frame rows")
        raise ValueError("Duplicate tracking entity-frame rows")

    origins = primary_origins.loc[
        primary_origins["eligible"].fillna(False)
    ].copy()
    origin_key = ["game_id", "play_id", "target_nfl_id", "origin_frame"]
    if origins.duplicated(origin_key).any():
        raise ValueError("Duplicate target origin rows")
    seeds = _explode_defenders(origins, "observed_origin_defender_ids")
    if seeds["target_nfl_id"].astype("string").eq(
        seeds["defender_nfl_id"].astype("string")
    ).any():
        raise ValueError("Target self-pairs are forbidden")

    input_tracking = tracking.loc[
        tracking["phase"].astype("string").eq("input")
    ]
    lookup = input_tracking[
        [
            "game_id",
            "play_id",
            "frame_id",
            "nfl_id",
            "player_side",
            "player_role",
            "x_norm",
            "y_norm",
            "normalized_coordinate_class",
        ]
    ]
    target_lookup = lookup.rename(
        columns={
            "frame_id": "origin_frame",
            "nfl_id": "target_nfl_id",
            "player_side": "target_side",
            "player_role": "target_role",
            "x_norm": "target_x",
            "y_norm": "target_y",
            "normalized_coordinate_class": "target_class",
        }
    )
    origin = seeds.merge(
        target_lookup,
        on=["game_id", "play_id", "origin_frame", "target_nfl_id"],
        how="left",
        validate="many_to_one",
        indicator="_target_merge",
    )
    unmatched_targets = int(origin["_target_merge"].ne("both").sum())
    defender_lookup = lookup.rename(
        columns={
            "frame_id": "origin_frame",
            "nfl_id": "defender_nfl_id",
            "player_side": "defender_side",
            "player_role": "defender_role",
            "x_norm": "defender_x",
            "y_norm": "defender_y",
            "normalized_coordinate_class": "defender_class",
        }
    )
    origin = origin.merge(
        defender_lookup,
        on=["game_id", "play_id", "origin_frame", "defender_nfl_id"],
        how="left",
        validate="many_to_one",
        indicator="_defender_merge",
    )
    unmatched_defenders = int(origin["_defender_merge"].ne("both").sum())
    matched = origin["_target_merge"].eq("both") & origin[
        "_defender_merge"
    ].eq("both")
    role_rows = origin.loc[matched]
    if not (
        role_rows["target_side"].astype("string").eq("Offense")
        & role_rows["target_role"]
        .astype("string")
        .eq("Targeted Receiver")
    ).all():
        raise ValueError("Unsupported target side or role")
    if not (
        role_rows["defender_side"].astype("string").eq("Defense")
        & role_rows["defender_role"]
        .astype("string")
        .eq("Defensive Coverage")
    ).all():
        raise ValueError("Unsupported defender side or role")
    valid_origin = (
        matched
        & _valid_coordinates(origin, "target")
        & _valid_coordinates(origin, "defender")
    )
    origin = origin.loc[valid_origin].copy()
    origin["dx"] = origin["defender_x"] - origin["target_x"]
    origin["dy"] = origin["defender_y"] - origin["target_y"]
    origin["separation_origin"] = np.hypot(origin["dx"], origin["dy"])
    origin_pairs = origin[
        [
            "game_id",
            "play_id",
            "week",
            "split",
            "target_nfl_id",
            "defender_nfl_id",
            "origin_frame",
            "target_x",
            "target_y",
            "defender_x",
            "defender_y",
            "dx",
            "dy",
            "separation_origin",
        ]
    ].sort_values(
        ["game_id", "play_id", "origin_frame", "target_nfl_id", "defender_nfl_id"],
        kind="stable",
    )

    future = future_separation_eligibility.loc[
        future_separation_eligibility["eligible"].fillna(False)
    ].copy()
    expected = _explode_defenders(future, "evaluable_defender_ids")
    expected = expected[
        [
            "game_id",
            "play_id",
            "target_nfl_id",
            "origin_frame",
            "horizon",
            "defender_nfl_id",
        ]
    ].copy()
    observed_counts = expected.groupby(
        ["game_id", "play_id", "target_nfl_id", "origin_frame", "horizon"]
    ).size()
    declared_counts = future.set_index(
        ["game_id", "play_id", "target_nfl_id", "origin_frame", "horizon"]
    )["evaluable_defender_count"]
    if not observed_counts.eq(declared_counts.loc[observed_counts.index]).all():
        raise ValueError("Evaluable defender declarations do not reconcile")

    candidate = expected.merge(
        origin_pairs,
        on=[
            "game_id",
            "play_id",
            "target_nfl_id",
            "defender_nfl_id",
            "origin_frame",
        ],
        how="left",
        validate="many_to_one",
        indicator="_origin_merge",
    )
    unavailable_origin = int(candidate["_origin_merge"].ne("both").sum())
    candidate = candidate.loc[candidate["_origin_merge"].eq("both")].copy()

    trajectories = trajectory_eligibility.loc[
        trajectory_eligibility["eligible"].fillna(False)
    ]
    target_trajectory = trajectories.loc[
        trajectories["is_target_role"].fillna(False),
        [
            "game_id",
            "play_id",
            "target_nfl_id",
            "origin_frame",
            "horizon",
            "label_phase",
            "label_end_frame",
        ],
    ].rename(
        columns={
            "label_phase": "target_future_phase",
            "label_end_frame": "target_future_frame",
        }
    )
    candidate = candidate.merge(
        target_trajectory,
        on=["game_id", "play_id", "target_nfl_id", "origin_frame", "horizon"],
        how="left",
        validate="many_to_one",
        indicator="_target_trajectory_merge",
    )
    unavailable_target_trajectory = int(
        candidate["_target_trajectory_merge"].ne("both").sum()
    )
    candidate = candidate.loc[
        candidate["_target_trajectory_merge"].eq("both")
    ].copy()
    defender_trajectory = trajectories.loc[
        trajectories["is_defender"].fillna(False),
        [
            "game_id",
            "play_id",
            "nfl_id",
            "origin_frame",
            "horizon",
            "label_phase",
            "label_end_frame",
        ],
    ].rename(
        columns={
            "nfl_id": "defender_nfl_id",
            "label_phase": "defender_future_phase",
            "label_end_frame": "defender_future_frame",
        }
    )
    candidate = candidate.merge(
        defender_trajectory,
        on=[
            "game_id",
            "play_id",
            "defender_nfl_id",
            "origin_frame",
            "horizon",
        ],
        how="left",
        validate="many_to_one",
        indicator="_defender_trajectory_merge",
    )
    unavailable_defender_trajectory = int(
        candidate["_defender_trajectory_merge"].ne("both").sum()
    )
    candidate = candidate.loc[
        candidate["_defender_trajectory_merge"].eq("both")
    ].copy()
    if not (
        candidate["target_future_phase"].astype("string").eq("output")
        & candidate["defender_future_phase"].astype("string").eq("output")
        & candidate["target_future_frame"].eq(candidate["horizon"])
        & candidate["defender_future_frame"].eq(candidate["horizon"])
    ).all():
        raise ValueError("Unsupported future phase or horizon mapping")

    output_lookup = tracking.loc[
        tracking["phase"].astype("string").eq("output"),
        [
            "game_id",
            "play_id",
            "frame_id",
            "nfl_id",
            "x_norm",
            "y_norm",
            "normalized_coordinate_class",
        ],
    ]
    target_future = output_lookup.rename(
        columns={
            "frame_id": "target_future_frame",
            "nfl_id": "target_nfl_id",
            "x_norm": "target_future_x",
            "y_norm": "target_future_y",
            "normalized_coordinate_class": "target_future_class",
        }
    )
    candidate = candidate.merge(
        target_future,
        on=["game_id", "play_id", "target_nfl_id", "target_future_frame"],
        how="left",
        validate="many_to_one",
        indicator="_target_future_merge",
    )
    unavailable_target_future = int(
        candidate["_target_future_merge"].ne("both").sum()
    )
    candidate = candidate.loc[candidate["_target_future_merge"].eq("both")].copy()
    defender_future = output_lookup.rename(
        columns={
            "frame_id": "defender_future_frame",
            "nfl_id": "defender_nfl_id",
            "x_norm": "defender_future_x",
            "y_norm": "defender_future_y",
            "normalized_coordinate_class": "defender_future_class",
        }
    )
    candidate = candidate.merge(
        defender_future,
        on=[
            "game_id",
            "play_id",
            "defender_nfl_id",
            "defender_future_frame",
        ],
        how="left",
        validate="many_to_one",
        indicator="_defender_future_merge",
    )
    unavailable_defender_future = int(
        candidate["_defender_future_merge"].ne("both").sum()
    )
    candidate = candidate.loc[
        candidate["_defender_future_merge"].eq("both")
    ].copy()
    valid_future = _valid_coordinates(candidate, "target_future") & (
        _valid_coordinates(candidate, "defender_future")
    )
    unavailable_future_coordinates = int((~valid_future).sum())
    candidate = candidate.loc[valid_future].copy()
    candidate["separation_future"] = np.hypot(
        candidate["defender_future_x"] - candidate["target_future_x"],
        candidate["defender_future_y"] - candidate["target_future_y"],
    )
    candidate["separation_change"] = (
        candidate["separation_future"] - candidate["separation_origin"]
    )
    nonfinite = int(
        (
            ~np.isfinite(candidate["separation_origin"])
            | ~np.isfinite(candidate["separation_future"])
            | candidate["separation_origin"].lt(0)
            | candidate["separation_future"].lt(0)
        ).sum()
    )
    candidate = candidate.loc[
        np.isfinite(candidate["separation_origin"])
        & np.isfinite(candidate["separation_future"])
        & candidate["separation_origin"].ge(0)
        & candidate["separation_future"].ge(0)
    ].copy()
    pairs = candidate.rename(
        columns={
            "target_x": "target_x_origin",
            "target_y": "target_y_origin",
            "defender_x": "defender_x_origin",
            "defender_y": "defender_y_origin",
            "target_future_x": "target_x_future",
            "target_future_y": "target_y_future",
            "defender_future_x": "defender_x_future",
            "defender_future_y": "defender_y_future",
        }
    )
    pairs = pairs[PAIR_COLUMNS].sort_values(PAIR_KEY, kind="stable").reset_index(
        drop=True
    )
    duplicate_pairs = int(pairs.duplicated(PAIR_KEY).sum())
    if duplicate_pairs:
        raise ValueError("Duplicate receiver-defender pair identities")
    expected_keys = set(map(tuple, expected[PAIR_KEY].to_numpy()))
    actual_keys = set(map(tuple, pairs[PAIR_KEY].to_numpy()))
    eligibility_mismatch = len(expected_keys ^ actual_keys)
    unavailable_future = {
        "origin_pair": unavailable_origin,
        "target_trajectory": unavailable_target_trajectory,
        "defender_trajectory": unavailable_defender_trajectory,
        "target_tracking": unavailable_target_future,
        "defender_tracking": unavailable_defender_future,
        "coordinate": unavailable_future_coordinates,
    }
    diagnostics = {
        "source_origins": int(len(origins)),
        "valid_target_origins": int(
            origin_pairs[["game_id", "play_id", "target_nfl_id", "origin_frame"]]
            .drop_duplicates()
            .shape[0]
        ),
        "origin_pair_count": int(len(origin_pairs)),
        "unmatched_origin_target_rows": unmatched_targets,
        "unmatched_origin_defender_rows": unmatched_defenders,
        "expected_horizon_pairs": int(len(expected_keys)),
        "constructed_horizon_pairs": int(len(pairs)),
        "eligibility_mismatch_count": eligibility_mismatch,
        "unavailable_future_counts": unavailable_future,
        "duplicate_pair_keys": duplicate_pairs,
        "nonfinite_distance_count": nonfinite,
    }
    return ReceiverDefenderPairResult(
        pairs=pairs,
        origin_pairs=origin_pairs.reset_index(drop=True),
        diagnostics=diagnostics,
    )
