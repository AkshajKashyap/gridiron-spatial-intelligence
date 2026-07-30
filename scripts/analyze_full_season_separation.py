"""Sequential full-season receiver-observed-defender separation summary."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.receiver_defender_pairs import (  # noqa: E402
    PAIR_KEY,
    build_receiver_defender_pairs,
)
from gridiron_spatial.separation_summary import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFENDER_COUNT_BUCKETS,
    HORIZONS,
    ORIGIN_SEPARATION_BUCKETS,
    SPLITS,
    grouped_separation_summary,
    play_aggregate_bootstrap,
    prepare_analysis_pairs,
    separation_metrics_from_arrays,
)


EXPECTED_WEEKS = tuple(f"2023_w{week:02d}" for week in range(1, 19))
ANALYSIS_FORMAT_VERSION = "full_season_separation_summary_v1"
NORMALIZED_ARTIFACT_VERSION = "1.0"
COHORT_ARTIFACT_VERSION = "1.0"
COORDINATE_TRANSFORM_VERSION = "nfl_common_direction_v1"
TRACKING_COLUMNS = [
    "game_id",
    "play_id",
    "phase",
    "frame_id",
    "nfl_id",
    "week",
    "split",
    "player_side",
    "player_role",
    "x_norm",
    "y_norm",
    "normalized_coordinate_class",
]


class _MetricStore:
    """Keep identity-free value chunks plus only unique play keys."""

    def __init__(self, group_columns: tuple[str, ...]) -> None:
        self.group_columns = group_columns
        self.groups: dict[tuple[Any, ...], dict[str, Any]] = {}

    def update(self, frame: pd.DataFrame) -> None:
        for group, selected in frame.groupby(
            list(self.group_columns), sort=True, observed=True
        ):
            key = group if isinstance(group, tuple) else (group,)
            values = self.groups.setdefault(
                key,
                {"origin": [], "future": [], "change": [], "plays": set()},
            )
            values["origin"].append(
                selected["separation_origin"].to_numpy(dtype=float, copy=True)
            )
            values["future"].append(
                selected["separation_future"].to_numpy(dtype=float, copy=True)
            )
            values["change"].append(
                selected["separation_change"].to_numpy(dtype=float, copy=True)
            )
            values["plays"].update(
                map(
                    tuple,
                    selected[["game_id", "play_id"]]
                    .drop_duplicates()
                    .to_numpy(),
                )
            )

    def records(self) -> list[dict[str, Any]]:
        records = []
        for key in sorted(
            self.groups,
            key=lambda values: tuple(str(value) for value in values),
        ):
            values = self.groups[key]
            record_key = {
                column: _python_scalar(value)
                for column, value in zip(self.group_columns, key)
            }
            records.append(
                {
                    **record_key,
                    **separation_metrics_from_arrays(
                        np.concatenate(values["origin"]),
                        np.concatenate(values["future"]),
                        np.concatenate(values["change"]),
                        unique_play_count=len(values["plays"]),
                    ),
                }
            )
        return records


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Artifact path must be inside the project: {path}") from error


def _validate_manifests(
    normalized_root: Path,
    cohort_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    normalized = _load_json(normalized_root / "manifest.json")
    cohort = _load_json(cohort_root / "manifest.json")
    expected = list(EXPECTED_WEEKS)
    if normalized.get("validation_status") != "PASS":
        raise ValueError("Normalized manifest validation is not PASS")
    if normalized.get("requested_weeks") != expected:
        raise ValueError("Normalized requested weeks are not exactly Weeks 01-18")
    if normalized.get("processed_weeks") != expected:
        raise ValueError("Normalized processed weeks are not exactly Weeks 01-18")
    if normalized.get("artifact_format_version") != NORMALIZED_ARTIFACT_VERSION:
        raise ValueError("Unsupported normalized artifact format version")
    if (
        normalized.get("coordinate_transform_version")
        != COORDINATE_TRANSFORM_VERSION
    ):
        raise ValueError("Unsupported coordinate transform version")
    if cohort.get("processed_weeks") != expected:
        raise ValueError("Frozen cohort weeks are not exactly Weeks 01-18")
    if cohort.get("artifact_format_version") != COHORT_ARTIFACT_VERSION:
        raise ValueError("Unsupported cohort artifact format version")
    if cohort.get("aggregate_reconciliation_status") != "PASS":
        raise ValueError("Frozen cohort reconciliation is not PASS")
    if cohort.get("split_validation_status") != "PASS":
        raise ValueError("Frozen cohort split validation is not PASS")
    partitions = normalized.get("partitions", [])
    if [partition.get("week") for partition in partitions] != expected:
        raise ValueError("Normalized manifest partitions are missing or reordered")
    if any(
        partition.get("reconciliation_status") != "PASS"
        for partition in partitions
    ):
        raise ValueError("A normalized partition reconciliation is not PASS")
    by_week = {partition["week"]: partition for partition in partitions}
    for week in EXPECTED_WEEKS:
        path = normalized_root / by_week[week]["relative_filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing normalized partition: {path}")
    for table in (
        "primary_origins",
        "trajectory_eligibility",
        "future_separation_eligibility",
    ):
        path = cohort_root / f"{table}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen cohort table: {path}")
    return normalized, cohort, by_week


def _read_cohort_week(root: Path, table: str, week: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        root / f"{table}.parquet",
        filters=[("week", "==", week)],
    )
    if frame.empty:
        raise ValueError(f"Frozen cohort table {table} has no rows for {week}")
    if set(frame["week"].astype("string")) != {week}:
        raise ValueError(f"Frozen cohort table {table} leaked another week")
    return frame


def _play_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(_closing=frame["separation_change"].lt(0).astype(float))
        .groupby(["split", "horizon", "game_id", "play_id"], sort=True)
        .agg(
            mean_separation_change=("separation_change", "mean"),
            closing_fraction=("_closing", "mean"),
        )
        .reset_index()
    )


def _ordered_results(
    records: list[dict[str, Any]],
    field: str,
    order: tuple[Any, ...],
) -> list[dict[str, Any]]:
    rank = {value: position for position, value in enumerate(order)}
    return sorted(records, key=lambda row: (rank[row[field]], row["horizon"]))


def _stability(
    weekly: list[dict[str, Any]],
    split: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for horizon in HORIZONS:
        weeks = [row for row in weekly if row["horizon"] == horizon]
        splits = [row for row in split if row["horizon"] == horizon]
        week_one = next(
            row for row in weeks if row["week"] == EXPECTED_WEEKS[0]
        )
        week_one_sign = int(np.sign(week_one["mean_separation_change"]))
        result.append(
            {
                "horizon": horizon,
                "week_01_mean_separation_change": week_one[
                    "mean_separation_change"
                ],
                "week_01_closing_fraction": week_one["closing_fraction"],
                "weeks_with_negative_mean_change": int(sum(
                    row["mean_separation_change"] < 0 for row in weeks
                )),
                "weeks_with_closing_fraction_above_half": int(sum(
                    row["closing_fraction"] > 0.5 for row in weeks
                )),
                "weeks_matching_week_01_mean_change_direction": int(sum(
                    int(np.sign(row["mean_separation_change"])) == week_one_sign
                    for row in weeks
                )),
                "splits_with_negative_mean_change": int(sum(
                    row["mean_separation_change"] < 0 for row in splits
                )),
                "splits_with_closing_fraction_above_half": int(sum(
                    row["closing_fraction"] > 0.5 for row in splits
                )),
                "week_count": len(weeks),
                "split_count": len(splits),
            }
        )
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _console_summary(result: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "pair_count",
        "unique_play_count",
        "mean_separation_change",
        "median_separation_change",
        "closing_fraction",
        "unchanged_fraction",
        "expanding_fraction",
    )

    def compact(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in record.items()
                if key in fields
                or key
                in {
                    "week",
                    "split",
                    "horizon",
                    "origin_separation_bucket",
                    "origin_defender_count_bucket",
                }
            }
            for record in records
        ]

    return {
        "output": result["output_artifact"],
        "total_runtime_seconds": result["total_runtime_seconds"],
        "runtime_by_week_seconds": result["runtime_by_week_seconds"],
        "aggregate_counts": result["aggregate_counts"],
        "split_horizon_results": compact(result["split_results"]),
        "play_cluster_bootstrap_results": result[
            "play_cluster_bootstrap_results"
        ],
        "week_to_week_stability": result["week_to_week_stability"],
        "origin_separation_bucket_findings": compact(
            result["origin_separation_bucket_results"]
        ),
        "defender_count_bucket_findings": compact(
            result["defender_count_bucket_results"]
        ),
        "nearest_observed_defender_findings": compact(
            result["nearest_observed_defender_results"]
        ),
        "validation": result["validation_diagnostics"],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    started = time.perf_counter()
    normalized_root = (PROJECT_ROOT / args.normalized_root).resolve()
    cohort_root = (PROJECT_ROOT / args.cohort_root).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    normalized, cohort, partitions = _validate_manifests(
        normalized_root, cohort_root
    )

    all_store = _MetricStore(("horizon",))
    split_store = _MetricStore(("split", "horizon"))
    origin_bucket_store = _MetricStore(
        ("origin_separation_bucket", "horizon")
    )
    defender_bucket_store = _MetricStore(
        ("origin_defender_count_bucket", "horizon")
    )
    nearest_store = _MetricStore(("horizon",))
    weekly_results: list[dict[str, Any]] = []
    play_summary_chunks: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}
    weekly_pair_counts: dict[str, dict[str, int]] = {}
    weekly_diagnostics: list[dict[str, Any]] = []
    game_ids: set[Any] = set()
    origin_play_ids: set[tuple[Any, Any]] = set()
    origin_pair_count = 0
    valid_target_origins = 0
    mismatch_count = 0

    for week in EXPECTED_WEEKS:
        week_started = time.perf_counter()
        tracking_path = normalized_root / partitions[week]["relative_filename"]
        tracking = pd.read_parquet(tracking_path, columns=TRACKING_COLUMNS)
        if set(tracking["week"].astype("string")) != {week}:
            raise ValueError(f"Normalized partition week mismatch: {week}")
        origins = _read_cohort_week(cohort_root, "primary_origins", week)
        trajectories = _read_cohort_week(
            cohort_root, "trajectory_eligibility", week
        )
        future = _read_cohort_week(
            cohort_root, "future_separation_eligibility", week
        )
        pair_result = build_receiver_defender_pairs(
            tracking, origins, trajectories, future
        )
        diagnostics = pair_result.diagnostics
        analysis = prepare_analysis_pairs(
            pair_result.pairs, pair_result.origin_pairs
        )
        if analysis.duplicated(PAIR_KEY).any():
            raise ValueError(f"Duplicate horizon pair keys in {week}")
        if set(analysis["horizon"].unique()) != set(HORIZONS):
            raise ValueError(f"Unsupported or missing horizon in {week}")
        if not set(analysis["split"].astype("string")).issubset(SPLITS):
            raise ValueError(f"Unsupported split in {week}")
        if diagnostics["eligibility_mismatch_count"]:
            raise ValueError(f"Eligibility mismatch in {week}")
        if diagnostics["nonfinite_distance_count"]:
            raise ValueError(f"Non-finite distance in {week}")
        origin_key = [
            "game_id",
            "play_id",
            "target_nfl_id",
            "defender_nfl_id",
            "origin_frame",
        ]
        if pair_result.origin_pairs.duplicated(origin_key).any():
            raise ValueError(f"Duplicate origin pair keys in {week}")

        all_store.update(analysis)
        split_store.update(analysis)
        origin_bucket_store.update(analysis)
        defender_bucket_store.update(analysis)
        nearest_store.update(
            analysis.loc[analysis["is_nearest_observed_defender"]]
        )
        week_summary = grouped_separation_summary(analysis, ["horizon"])
        if len(week_summary) != len(HORIZONS):
            raise ValueError(f"Weekly analysis was skipped for {week}")
        weekly_results.extend(
            [{"week": week, **record} for record in week_summary]
        )
        play_summary_chunks.append(_play_summaries(analysis))
        counts = {
            str(horizon): int(analysis["horizon"].eq(horizon).sum())
            for horizon in HORIZONS
        }
        weekly_pair_counts[week] = counts
        game_ids.update(pair_result.origin_pairs["game_id"].unique())
        origin_play_ids.update(
            map(
                tuple,
                pair_result.origin_pairs[["game_id", "play_id"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        origin_pair_count += diagnostics["origin_pair_count"]
        valid_target_origins += diagnostics["valid_target_origins"]
        mismatch_count += diagnostics["eligibility_mismatch_count"]
        runtimes[week] = round(time.perf_counter() - week_started, 3)
        weekly_diagnostics.append(
            {
                "week": week,
                "origin_pair_count": diagnostics["origin_pair_count"],
                "expected_horizon_pairs": diagnostics[
                    "expected_horizon_pairs"
                ],
                "constructed_horizon_pairs": diagnostics[
                    "constructed_horizon_pairs"
                ],
                "eligibility_mismatch_count": diagnostics[
                    "eligibility_mismatch_count"
                ],
                "duplicate_pair_keys": diagnostics["duplicate_pair_keys"],
                "nonfinite_distance_count": diagnostics[
                    "nonfinite_distance_count"
                ],
            }
        )
        del (
            tracking,
            origins,
            trajectories,
            future,
            pair_result,
            analysis,
        )
        gc.collect()

    all_results = all_store.records()
    split_results = _ordered_results(
        split_store.records(), "split", SPLITS
    )
    origin_bucket_results = _ordered_results(
        origin_bucket_store.records(),
        "origin_separation_bucket",
        ORIGIN_SEPARATION_BUCKETS,
    )
    defender_bucket_results = _ordered_results(
        defender_bucket_store.records(),
        "origin_defender_count_bucket",
        DEFENDER_COUNT_BUCKETS,
    )
    nearest_results = nearest_store.records()
    weekly_results.sort(
        key=lambda row: (EXPECTED_WEEKS.index(row["week"]), row["horizon"])
    )
    play_summaries = pd.concat(play_summary_chunks, ignore_index=True)
    bootstrap_results = []
    for split in SPLITS:
        for horizon in HORIZONS:
            selected = play_summaries.loc[
                play_summaries["split"].astype("string").eq(split)
                & play_summaries["horizon"].eq(horizon)
            ]
            if selected.empty:
                raise ValueError(f"Empty bootstrap group: {split}/{horizon}")
            bootstrap_results.append(
                {
                    "split": split,
                    "horizon": horizon,
                    **play_aggregate_bootstrap(selected),
                }
            )

    expected_total = sum(
        sum(counts.values()) for counts in weekly_pair_counts.values()
    )
    aggregate_total = sum(row["pair_count"] for row in all_results)
    summarized_weekly_counts = {
        week: {
            str(horizon): int(
                next(
                    row["pair_count"]
                    for row in weekly_results
                    if row["week"] == week and row["horizon"] == horizon
                )
            )
            for horizon in HORIZONS
        }
        for week in EXPECTED_WEEKS
    }
    weekly_reconciliation = {
        "status": (
            "PASS"
            if summarized_weekly_counts == weekly_pair_counts
            else "FAIL"
        ),
        "expected": weekly_pair_counts,
        "observed": summarized_weekly_counts,
    }
    if expected_total != aggregate_total:
        raise ValueError("Aggregate pair counts do not equal weekly counts")
    if weekly_reconciliation["status"] != "PASS":
        raise ValueError("Weekly pair-count reconciliation failed")
    observed_splits = {
        record["split"] for record in split_results
    }
    if observed_splits != set(SPLITS):
        raise ValueError("Not all predetermined splits were analyzed")

    total_runtime = round(time.perf_counter() - started, 3)
    aggregate_counts = {
        "game_count": len(game_ids),
        "play_count": len(origin_play_ids),
        "valid_target_origin_count": valid_target_origins,
        "origin_pair_count": origin_pair_count,
        "horizon_pair_count": aggregate_total,
        "horizon_pair_counts": {
            str(row["horizon"]): row["pair_count"] for row in all_results
        },
        "horizon_unique_play_counts": {
            str(row["horizon"]): row["unique_play_count"]
            for row in all_results
        },
        "nearest_observed_defender_pair_counts": {
            str(row["horizon"]): row["pair_count"]
            for row in nearest_results
        },
    }
    result = {
        "analysis_format_version": ANALYSIS_FORMAT_VERSION,
        "generation_timestamp_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "output_artifact": _project_relative(output),
        "source_artifacts": {
            "normalized_tracking_root": _project_relative(normalized_root),
            "cohort_root": _project_relative(cohort_root),
        },
        "source_artifact_versions": {
            "normalized_tracking_artifact_format": normalized[
                "artifact_format_version"
            ],
            "coordinate_transform": normalized[
                "coordinate_transform_version"
            ],
            "cohort_artifact_format": cohort["artifact_format_version"],
        },
        "processed_weeks": list(EXPECTED_WEEKS),
        "fixed_definitions": {
            "horizons_frames": list(HORIZONS),
            "separation_change": (
                "separation_future - separation_origin"
            ),
            "change_classes": {
                "closing": "< 0",
                "unchanged": "== 0",
                "expanding": "> 0",
            },
            "origin_separation_buckets_yards": list(
                ORIGIN_SEPARATION_BUCKETS
            ),
            "origin_defender_count_buckets": list(DEFENDER_COUNT_BUCKETS),
            "nearest_observed_defender": (
                "Minimum origin separation; ties ordered by defender "
                "identifier. Descriptive reduction only, not an official "
                "coverage assignment."
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "play",
        },
        "claim_boundary": [
            "Analysis is conditioned on the competition-designated target.",
            "Defenders are observed defensive entities.",
            "Nearest observed defender is not an official coverage assignment.",
            "The analysis does not evaluate QB target selection.",
            "The analysis does not establish complete passing-window openness.",
            "The analysis does not establish causality.",
            "The analysis does not yet evaluate predictive performance.",
        ],
        "aggregate_counts": aggregate_counts,
        "validation_diagnostics": {
            "status": "PASS",
            "normalized_manifest_validation": "PASS",
            "cohort_manifest_reconciliation": "PASS",
            "cohort_manifest_split_validation": "PASS",
            "processed_week_count": len(EXPECTED_WEEKS),
            "eligibility_mismatch_count": mismatch_count,
            "aggregate_pair_count_reconciliation": {
                "status": "PASS",
                "summed_weekly_pairs": expected_total,
                "aggregate_pairs": aggregate_total,
            },
            "weekly_pair_count_reconciliation": weekly_reconciliation,
            "weekly_diagnostics": weekly_diagnostics,
        },
        "all_pair_results": all_results,
        "split_results": split_results,
        "weekly_results": weekly_results,
        "origin_separation_bucket_results": origin_bucket_results,
        "defender_count_bucket_results": defender_bucket_results,
        "nearest_observed_defender_results": nearest_results,
        "play_cluster_bootstrap_results": bootstrap_results,
        "week_to_week_stability": _stability(
            weekly_results, split_results
        ),
        "runtime_by_week_seconds": runtimes,
        "total_runtime_seconds": total_runtime,
    }
    _atomic_json(output, result)
    print(json.dumps(_console_summary(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
