"""Read-only Week 01 smoke test for the analytic cohort builder."""

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
WEEK = "2023_w01"


def _read_csv(path: Path, usecols: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype={"game_id": "string", "play_id": "string", "nfl_id": "string"},
        low_memory=False,
    )


def main() -> int:
    started = time.perf_counter()
    input_path = DATASET_ROOT / "train" / f"input_{WEEK}.csv"
    output_path = DATASET_ROOT / "train" / f"output_{WEEK}.csv"
    supplementary_path = DATASET_ROOT / "supplementary_data.csv"
    for path in (input_path, output_path, supplementary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required smoke-test input is missing: {path}")

    inputs = _read_csv(input_path, INPUT_USECOLS)
    outputs = _read_csv(output_path, OUTPUT_USECOLS)
    supplementary = pd.read_csv(
        supplementary_path,
        usecols=["game_id", "play_id", "week"],
        dtype={"game_id": "string", "play_id": "string"},
        low_memory=False,
    )

    cohorts = build_week_cohorts(inputs, outputs, supplementary, WEEK)
    tables = {
        name: getattr(cohorts, name) for name in COHORT_TABLE_NAMES
    }
    table_counts: dict[str, dict[str, int]] = {}
    for name, table in tables.items():
        if list(table.columns) != TABLE_SCHEMAS[name]:
            raise RuntimeError(f"Frozen schema mismatch for {name}")
        if table.duplicated(TABLE_KEY_COLUMNS[name]).any():
            raise RuntimeError(f"Duplicate frozen keys in {name}")
        eligible = int(table["eligible"].sum())
        table_counts[name] = {
            "rows": int(len(table)),
            "eligible": eligible,
            "excluded": int(len(table) - eligible),
        }

    ledger = build_exclusion_ledger(
        *(tables[name] for name in COHORT_TABLE_NAMES)
    )
    reconciliation = _table_reconciliation(tables, ledger)
    split_validation = _validate_game_splits(tables)

    horizon_counts: dict[str, dict[str, dict[str, int]]] = {}
    for table_name in (
        "trajectory_eligibility",
        "future_separation_eligibility",
    ):
        table = tables[table_name]
        horizon_counts[table_name] = {}
        for horizon in HORIZONS:
            selected = table.loc[table["horizon"].eq(horizon)]
            horizon_counts[table_name][str(horizon)] = {
                "rows": int(len(selected)),
                "eligible": int(selected["eligible"].sum()),
                "excluded": int((~selected["eligible"]).sum()),
            }

    descriptive = tables["descriptive_target_frames"]
    summary = {
        "week": WEEK,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "raw_rows": {
            "input": int(len(inputs)),
            "output": int(len(outputs)),
            "supplementary": int(len(supplementary)),
        },
        "cohort_tables": table_counts,
        "exclusion_ledger_rows": int(len(ledger)),
        "reconciliation_status": reconciliation["overall"][
            "reconciliation_status"
        ],
        "split_validation_status": split_validation["status"],
        "horizon_counts": horizon_counts,
        "descriptive_target_frame_count": int(len(descriptive)),
        "zero_defender_exclusion_count": int(descriptive["C07"].sum()),
    }
    print(json.dumps(summary, indent=2, sort_keys=False))
    if (
        summary["reconciliation_status"] != "PASS"
        or summary["split_validation_status"] != "PASS"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
