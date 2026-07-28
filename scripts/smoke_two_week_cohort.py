"""Artifact-free Weeks 01–02 aggregation smoke test."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gridiron_spatial.cohort import (  # noqa: E402
    COHORT_TABLE_NAMES,
    HORIZONS,
    INPUT_USECOLS,
    OUTPUT_USECOLS,
    TABLE_KEY_COLUMNS,
    TABLE_SCHEMAS,
    _table_reconciliation,
    _validate_game_splits,
    build_exclusion_ledger,
    build_week_cohorts,
)


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bdb_2026"
    / "114239_nfl_competition_files_published_analytics_final"
)
WEEKS = ("2023_w01", "2023_w02")


def _read_tracking(path: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype={"game_id": "string", "play_id": "string", "nfl_id": "string"},
        low_memory=False,
    )


def _counts(table: pd.DataFrame) -> dict[str, int]:
    eligible = int(table["eligible"].sum())
    return {
        "rows": int(len(table)),
        "eligible": eligible,
        "excluded": int(len(table) - eligible),
    }


def _validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    duplicate_counts: dict[str, int] = {}
    for name in COHORT_TABLE_NAMES:
        table = tables[name]
        if list(table.columns) != TABLE_SCHEMAS[name]:
            raise RuntimeError(f"Frozen schema mismatch for {name}")
        duplicates = int(table.duplicated(TABLE_KEY_COLUMNS[name]).sum())
        duplicate_counts[name] = duplicates
        if duplicates:
            raise RuntimeError(f"Duplicate frozen keys in {name}: {duplicates}")
    return duplicate_counts


def main() -> int:
    total_started = time.perf_counter()
    supplementary_source = pd.read_csv(
        DATASET_ROOT / "supplementary_data.csv",
        usecols=["game_id", "play_id", "week"],
        dtype={"game_id": "string", "play_id": "string"},
        low_memory=False,
    )
    weekly: dict[str, dict[str, object]] = {}

    for week in WEEKS:
        week_started = time.perf_counter()
        inputs = _read_tracking(
            DATASET_ROOT / "train" / f"input_{week}.csv",
            INPUT_USECOLS,
        )
        outputs = _read_tracking(
            DATASET_ROOT / "train" / f"output_{week}.csv",
            OUTPUT_USECOLS,
        )
        play_keys = inputs[["game_id", "play_id"]].drop_duplicates()
        supplementary = supplementary_source.merge(
            play_keys,
            on=["game_id", "play_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(supplementary) != len(play_keys):
            raise RuntimeError(
                f"{week} supplementary coverage mismatch: "
                f"{len(supplementary)} != {len(play_keys)}"
            )

        cohorts = build_week_cohorts(inputs, outputs, supplementary, week)
        tables = {
            name: getattr(cohorts, name) for name in COHORT_TABLE_NAMES
        }
        _validate_tables(tables)
        ledger = build_exclusion_ledger(
            *(tables[name] for name in COHORT_TABLE_NAMES)
        )
        reconciliation = _table_reconciliation(tables, ledger)
        status = reconciliation["overall"]["reconciliation_status"]
        runtime = round(time.perf_counter() - week_started, 3)
        print(f"{week} runtime_seconds={runtime:.3f} status={status}")
        if status != "PASS":
            return 1
        weekly[week] = {
            "runtime_seconds": runtime,
            "raw_rows": {
                "input": int(len(inputs)),
                "output": int(len(outputs)),
                "supplementary": int(len(supplementary)),
            },
            "tables": tables,
            "table_counts": {
                name: _counts(tables[name]) for name in COHORT_TABLE_NAMES
            },
            "ledger": ledger,
            "ledger_rows": int(len(ledger)),
            "reconciliation_status": status,
        }

    if tuple(weekly) != WEEKS:
        raise RuntimeError(f"Unexpected processed weeks: {tuple(weekly)}")

    aggregate_tables = {
        name: pd.concat(
            [weekly[week]["tables"][name] for week in WEEKS],
            ignore_index=True,
        )
        for name in COHORT_TABLE_NAMES
    }
    duplicate_key_counts = _validate_tables(aggregate_tables)
    aggregate_ledger = pd.concat(
        [weekly[week]["ledger"] for week in WEEKS],
        ignore_index=True,
    )
    duplicate_ledger_ids = int(
        aggregate_ledger["ledger_id"].duplicated().sum()
    )
    if duplicate_ledger_ids:
        raise RuntimeError(
            f"Duplicate aggregate ledger IDs: {duplicate_ledger_ids}"
        )

    aggregate_counts = {
        name: _counts(aggregate_tables[name]) for name in COHORT_TABLE_NAMES
    }
    summed_weekly_totals_match = True
    for name in COHORT_TABLE_NAMES:
        for metric in ("rows", "eligible", "excluded"):
            weekly_sum = sum(
                weekly[week]["table_counts"][name][metric] for week in WEEKS
            )
            if aggregate_counts[name][metric] != weekly_sum:
                summed_weekly_totals_match = False
    if not summed_weekly_totals_match:
        raise RuntimeError("Aggregate table totals do not equal weekly sums")

    aggregate_reconciliation = _table_reconciliation(
        aggregate_tables, aggregate_ledger
    )
    split_validation = _validate_game_splits(aggregate_tables)
    if aggregate_reconciliation["overall"]["reconciliation_status"] != "PASS":
        return 1
    if split_validation["status"] != "PASS":
        return 1

    horizon_counts: dict[str, dict[str, dict[str, int]]] = {}
    for table_name in (
        "trajectory_eligibility",
        "future_separation_eligibility",
    ):
        table = aggregate_tables[table_name]
        horizon_counts[table_name] = {
            str(horizon): _counts(
                table.loc[table["horizon"].eq(horizon)]
            )
            for horizon in HORIZONS
        }

    summary = {
        "weeks": list(WEEKS),
        "weekly": {
            week: {
                "runtime_seconds": weekly[week]["runtime_seconds"],
                "raw_rows": weekly[week]["raw_rows"],
                "table_counts": weekly[week]["table_counts"],
                "ledger_rows": weekly[week]["ledger_rows"],
                "reconciliation_status": weekly[week][
                    "reconciliation_status"
                ],
            }
            for week in WEEKS
        },
        "total_runtime_seconds": round(
            time.perf_counter() - total_started, 3
        ),
        "aggregate_table_counts": aggregate_counts,
        "aggregate_ledger_rows": int(len(aggregate_ledger)),
        "aggregate_reconciliation_status": aggregate_reconciliation[
            "overall"
        ]["reconciliation_status"],
        "aggregate_split_validation_status": split_validation["status"],
        "aggregate_duplicate_key_counts": duplicate_key_counts,
        "aggregate_duplicate_ledger_id_count": duplicate_ledger_ids,
        "horizon_counts": horizon_counts,
        "aggregate_totals_equal_summed_weekly_totals": (
            summed_weekly_totals_match
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
