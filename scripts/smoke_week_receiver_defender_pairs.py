"""Read-only Week 01 receiver-observed-defender separation smoke analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.normalized_tracking import (  # noqa: E402
    NORMALIZED_ENTITY_FRAME_KEY,
)
from gridiron_spatial.receiver_defender_pairs import (  # noqa: E402
    PAIR_KEY,
    build_receiver_defender_pairs,
)


WEEK = "2023_w01"
NORMALIZED_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "milestone_2"
    / "normalized_tracking"
    / f"normalized_{WEEK}.parquet"
)
COHORT_ROOT = PROJECT_ROOT / "artifacts" / "milestone_2" / "cohorts"


def _summary(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def _read_week_cohort(name: str) -> pd.DataFrame:
    path = COHORT_ROOT / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort artifact: {path}")
    return pd.read_parquet(path, filters=[("week", "==", WEEK)])


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True, choices=[WEEK])
    parser.parse_args(arguments)
    started = time.perf_counter()
    if not NORMALIZED_PATH.is_file():
        raise FileNotFoundError(
            f"Missing normalized Week 01 artifact: {NORMALIZED_PATH}"
        )

    tracking = pd.read_parquet(NORMALIZED_PATH)
    origins = _read_week_cohort("primary_origins")
    trajectories = _read_week_cohort("trajectory_eligibility")
    future = _read_week_cohort("future_separation_eligibility")
    result = build_receiver_defender_pairs(
        tracking, origins, trajectories, future
    )
    pairs = result.pairs
    origin_pairs = result.origin_pairs
    diagnostics = result.diagnostics

    defenders_per_play = (
        origin_pairs.groupby(["game_id", "play_id"])[
            "defender_nfl_id"
        ]
        .nunique()
        .value_counts()
        .sort_index()
    )
    horizon_pair_counts = {
        str(horizon): int(pairs["horizon"].eq(horizon).sum())
        for horizon in (5, 10, 15)
    }
    horizon_play_counts = {
        str(horizon): int(
            pairs.loc[
                pairs["horizon"].eq(horizon), ["game_id", "play_id"]
            ]
            .drop_duplicates()
            .shape[0]
        )
        for horizon in (5, 10, 15)
    }
    future_summaries = {
        str(horizon): _summary(
            pairs.loc[
                pairs["horizon"].eq(horizon), "separation_future"
            ]
        )
        for horizon in (5, 10, 15)
    }
    change_summaries = {
        str(horizon): _summary(
            pairs.loc[
                pairs["horizon"].eq(horizon), "separation_change"
            ]
        )
        for horizon in (5, 10, 15)
    }
    fraction_negative = {}
    fraction_positive = {}
    for horizon in (5, 10, 15):
        changes = pairs.loc[
            pairs["horizon"].eq(horizon), "separation_change"
        ]
        fraction_negative[str(horizon)] = (
            float(changes.lt(0).mean()) if len(changes) else None
        )
        fraction_positive[str(horizon)] = (
            float(changes.gt(0).mean()) if len(changes) else None
        )

    summary = {
        "week": WEEK,
        "games": int(tracking["game_id"].nunique()),
        "plays": int(
            tracking[["game_id", "play_id"]].drop_duplicates().shape[0]
        ),
        "plays_with_valid_target_origin": diagnostics[
            "valid_target_origins"
        ],
        "origin_target_defender_pairs": diagnostics["origin_pair_count"],
        "unique_targets": int(origin_pairs["target_nfl_id"].nunique()),
        "unique_defenders": int(origin_pairs["defender_nfl_id"].nunique()),
        "defenders_per_play_distribution": {
            str(count): int(plays)
            for count, plays in defenders_per_play.items()
        },
        "pair_counts_by_horizon": horizon_pair_counts,
        "evaluable_plays_by_horizon": horizon_play_counts,
        "origin_separation": _summary(origin_pairs["separation_origin"]),
        "future_separation_by_horizon": future_summaries,
        "separation_change_by_horizon": change_summaries,
        "fraction_negative_change_by_horizon": fraction_negative,
        "fraction_positive_change_by_horizon": fraction_positive,
        "duplicate_key_counts": {
            "tracking": int(
                tracking.duplicated(NORMALIZED_ENTITY_FRAME_KEY).sum()
            ),
            "origin_pairs": int(
                origin_pairs.duplicated(
                    [
                        "game_id",
                        "play_id",
                        "target_nfl_id",
                        "defender_nfl_id",
                        "origin_frame",
                    ]
                ).sum()
            ),
            "horizon_pairs": int(pairs.duplicated(PAIR_KEY).sum()),
        },
        "unmatched_origin_counts": {
            "target_rows": diagnostics["unmatched_origin_target_rows"],
            "defender_rows": diagnostics[
                "unmatched_origin_defender_rows"
            ],
        },
        "unavailable_future_counts": diagnostics[
            "unavailable_future_counts"
        ],
        "nonfinite_distance_count": diagnostics[
            "nonfinite_distance_count"
        ],
        "eligibility_reconciliation": {
            "expected_pairs": diagnostics["expected_horizon_pairs"],
            "constructed_pairs": diagnostics[
                "constructed_horizon_pairs"
            ],
            "mismatch_count": diagnostics[
                "eligibility_mismatch_count"
            ],
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(summary, indent=2))
    failures = [
        any(summary["duplicate_key_counts"].values()),
        diagnostics["eligibility_mismatch_count"] != 0,
        diagnostics["nonfinite_distance_count"] != 0,
    ]
    return 1 if any(failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
